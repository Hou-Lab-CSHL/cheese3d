"""SLEAP 1.6 single-instance backend for Cheese3D."""

import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from cheese3d.backends.core import (
    Pose2dBackend,
    isolate_worker_output,
    monitor_camera_progress,
    partition_videos_by_gpu,
    read_foreign_records,
    register_pose_backend,
    shutdown_completed_process_pool,
)
from cheese3d.config import KeypointConfig
from cheese3d.utils import BoundingBox


SLEAP_BACKBONES = (
    "unet", "unet_medium_rf", "unet_large_rf",
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    "swint_tiny", "swint_small", "swint_base",
)


def _make_skeleton(keypoint_names: List[str], edges: List[List[str]]):
    """Construct a SLEAP skeleton while ignoring edges absent from this view set."""
    import sleap_io as sio

    skeleton = sio.Skeleton(name="cheese3d")
    skeleton.add_nodes(keypoint_names)
    names = set(keypoint_names)
    for edge in edges:
        if len(edge) == 2 and edge[0] in names and edge[1] in names:
            skeleton.add_edge(edge[0], edge[1])
    return skeleton


def _annotation_records(label_dirs: Dict[str, Path], keypoint_names: List[str]) -> dict:
    """Read Cheese3D annotations into image-folder records used to build SLP."""
    records = {}
    for folder_name, folder in label_dirs.items():
        annotation_path = Path(folder) / "annotations.yaml"
        if not annotation_path.is_file():
            continue
        annotations = yaml.safe_load(annotation_path.read_text()) or {}
        image_names = sorted({
            image_name
            for keypoint in keypoint_names
            for image_name in (annotations.get(keypoint) or {})
            if (Path(folder) / image_name).is_file()
        })
        for image_name in image_names:
            points = []
            for keypoint in keypoint_names:
                values = (annotations.get(keypoint) or {}).get(image_name) or []
                point = values[0] if values and values[0] else [None, None]
                points.append([
                    np.nan if point[0] is None else float(point[0]),
                    np.nan if point[1] is None else float(point[1]),
                ])
            records[(folder_name, image_name)] = (Path(folder) / image_name, points)
    return records


def create_sleap_labels(records: dict, output_path: Path, keypoint_names: List[str],
                        skeleton_edges: List[List[str]]) -> int:
    """Write portable image-sequence SLEAP labels from normalized records."""
    import sleap_io as sio

    skeleton = _make_skeleton(keypoint_names, skeleton_edges)
    labeled_frames = []
    videos = []
    by_folder = {}
    for (folder_name, image_name), value in records.items():
        by_folder.setdefault(folder_name, []).append((image_name, value))
    for folder_name, items in sorted(by_folder.items()):
        items.sort(key=lambda item: item[0])
        filenames = [str(value[0]) for _, value in items]
        # Without an explicit value, sleap_io autodetects grayscale-vs-RGB
        # independently per video from just its first frame. Cheese3D always
        # stores frames as RGB PNGs, but different sessions' first frames can
        # look incidentally monochrome, so autodetection produces videos with
        # inconsistent channel counts; a training batch mixing samples from
        # two such videos then crashes in collation. Force RGB uniformly here;
        # SLEAP-NN's own ensure_grayscale/ensure_rgb config still controls any
        # actual grayscale conversion, applied consistently across all frames.
        video = sio.Video.from_filename(filenames, grayscale=False)
        videos.append(video)
        for frame_idx, (_, (_, points)) in enumerate(items):
            instance = sio.Instance.from_numpy(np.asarray(points, dtype=float), skeleton)
            labeled_frames.append(sio.LabeledFrame(
                video=video, frame_idx=frame_idx, instances=[instance]
            ))
    labels = sio.Labels(
        labeled_frames=labeled_frames, videos=videos, skeletons=[skeleton]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(labels, str(output_path), embed=False, verbose=False)
    return len(labeled_frames)


def create_sleap_training_config(project_path: Path, model_name: str,
                                 keypoint_names: List[str]) -> Path:
    """Create the SLEAP-NN single-instance configuration used by the GUI."""
    from omegaconf import OmegaConf
    from sleap_nn.config.data_config import (
        AugmentationConfig, DataConfig, GeometricConfig, PreprocessingConfig,
    )
    from sleap_nn.config.model_config import (
        BackboneConfig, HeadConfig, ModelConfig, SingleInstanceConfig,
        SingleInstanceConfMapsConfig, UNetMediumRFConfig,
    )
    from sleap_nn.config.trainer_config import TrainerConfig
    from sleap_nn.config.training_job_config import TrainingJobConfig

    config = TrainingJobConfig(
        name=model_name,
        data_config=DataConfig(
            train_labels_path=[str(project_path / "labels.slp")],
            validation_fraction=0.1,
            # Cheese3D videos are RGB throughout (DLC and Lightning Pose never
            # force grayscale); matching that here also avoids a SLEAP-NN 0.1.0
            # validation-path bug where ensure_grayscale=True intermittently
            # feeds the model a 4-channel batch instead of the expected 1.
            preprocessing=PreprocessingConfig(ensure_grayscale=False, ensure_rgb=True),
            augmentation_config=AugmentationConfig(geometric=GeometricConfig()),
        ),
        model_config=ModelConfig(
            backbone_config=BackboneConfig(unet=UNetMediumRFConfig()),
            head_configs=HeadConfig(single_instance=SingleInstanceConfig(
                confmaps=SingleInstanceConfMapsConfig(
                    # SLEAP-NN defaults output_stride to 1 -- confidence maps at
                    # full input resolution -- which suits the small cropped
                    # frames SLEAP is usually pointed at. Cheese3D feeds whole
                    # 640x512 camera views with ~28 keypoints, where the decoder
                    # upsampling to stride 1 dominates memory: training OOMed on
                    # 47 GiB GPUs at every batch size tried (64, 32 and 16 per
                    # GPU), always inside torch's `interpolate`. Stride 2 cuts
                    # those decoder activations ~4x, which is what makes a
                    # useful batch size fit at all. Peak positions stay
                    # sub-pixel because SLEAP refines each peak locally, and
                    # DLC and Lightning Pose likewise predict strided heatmaps.
                    part_names=keypoint_names, sigma=5.0, output_stride=2,
                )
            )),
        ),
        trainer_config=TrainerConfig(
            save_ckpt=True, ckpt_dir=str(project_path / "models"),
            run_name=model_name, max_epochs=100,
        ),
    )
    config_path = project_path / "config.yaml"
    OmegaConf.save(config.to_sleap_nn_cfg(), config_path)
    return config_path


def _set_sleap_backbone(cfg, backbone: str) -> None:
    """Replace the one-of SLEAP backbone configuration from a GUI preset."""
    from omegaconf import OmegaConf
    from sleap_nn.config.model_config import (
        BackboneConfig, ConvNextBaseConfig, ConvNextConfig, ConvNextLargeConfig,
        ConvNextSmallConfig, SwinTBaseConfig, SwinTConfig, SwinTSmallConfig,
        UNetConfig, UNetLargeRFConfig, UNetMediumRFConfig,
    )

    factories = {
        "unet": lambda: BackboneConfig(unet=UNetConfig()),
        "unet_medium_rf": lambda: BackboneConfig(unet=UNetMediumRFConfig()),
        "unet_large_rf": lambda: BackboneConfig(unet=UNetLargeRFConfig()),
        "convnext_tiny": lambda: BackboneConfig(convnext=ConvNextConfig()),
        "convnext_small": lambda: BackboneConfig(convnext=ConvNextSmallConfig()),
        "convnext_base": lambda: BackboneConfig(convnext=ConvNextBaseConfig()),
        "convnext_large": lambda: BackboneConfig(convnext=ConvNextLargeConfig()),
        "swint_tiny": lambda: BackboneConfig(swint=SwinTConfig()),
        "swint_small": lambda: BackboneConfig(swint=SwinTSmallConfig()),
        "swint_base": lambda: BackboneConfig(swint=SwinTBaseConfig()),
    }
    if backbone not in factories:
        raise ValueError(f"Unsupported SLEAP backbone: {backbone}")
    backbone_config = OmegaConf.structured(factories[backbone]())
    _apply_sleap_pretrained_weights(backbone_config, backbone)
    _align_sleap_backbone_output_stride(backbone_config, cfg)
    cfg.model_config.backbone_config = backbone_config


# SLEAP-NN leaves `pre_trained_weights` at None, so every backbone trains from
# random initialization. UNet cannot be pretrained at all, and Cheese3D-sized
# datasets are small (a few hundred labeled frames per project), where a
# from-scratch model reliably collapses to the trivial "predict background
# everywhere" solution: confirmed by direct reproduction with unet_medium_rf on
# 860 images, whose loss plateaued from epoch 19 onward and whose inference
# placed 97.7% of predicted points on the image border with peak scores of
# ~0.035 -- below SLEAP's own 0.2 detection threshold, so every downstream
# triangulation input was NaN. ImageNet-pretrained ConvNext/SwinT backbones
# avoid that the same way DLC's pretrained ResNet does.
SLEAP_PRETRAINED_WEIGHTS = {
    "convnext_tiny": "ConvNeXt_Tiny_Weights",
    "convnext_small": "ConvNeXt_Small_Weights",
    "convnext_base": "ConvNeXt_Base_Weights",
    "convnext_large": "ConvNeXt_Large_Weights",
    "swint_tiny": "Swin_T_Weights",
    "swint_small": "Swin_S_Weights",
    "swint_base": "Swin_B_Weights",
}


def _align_sleap_backbone_output_stride(backbone_config, cfg) -> None:
    """Make the backbone decoder stop upsampling below the head's stride.

    The backbone and the confidence-map head each carry their own
    ``output_stride``, and selecting a backbone preset resets the backbone's
    to SLEAP-NN's default of 1. The decoder then upsamples all the way from
    ``max_stride`` 32 to full input resolution, which is what actually
    exhausts GPU memory on Cheese3D's whole-frame 640x512 views -- training
    OOMed inside torch's ``interpolate`` at 64, 32 and 16 samples per GPU
    alike, and setting only the head's stride did not help because the
    backbone kept producing full-resolution features regardless.
    """
    head_stride = 2
    try:
        head_stride = int(
            cfg.model_config.head_configs.single_instance.confmaps.output_stride
        )
    except Exception:
        pass
    for section in ("unet", "convnext", "swint"):
        block = getattr(backbone_config, section, None)
        if block is not None and hasattr(block, "output_stride"):
            block.output_stride = head_stride


def _apply_sleap_pretrained_weights(backbone_config, backbone: str) -> None:
    """Request ImageNet weights for backbones that support them.

    UNet variants have no ``pre_trained_weights`` field, so they are skipped
    and remain randomly initialized -- SLEAP-NN offers no pretrained UNet.
    """
    weights = SLEAP_PRETRAINED_WEIGHTS.get(backbone)
    if weights is None:
        return
    for section in ("convnext", "swint"):
        block = getattr(backbone_config, section, None)
        if block is not None:
            block.pre_trained_weights = weights


def _sleap_predictions_to_h5(slp_path: Path, h5_path: Path, scorer: str) -> Path:
    """Preserve SLEAP per-node scores in Cheese3D's DLC-compatible HDF5."""
    import sleap_io as sio

    labels = sio.load_slp(str(slp_path), open_videos=False)
    keypoint_names = [node.name for node in labels.skeletons[0].nodes]
    columns = pd.MultiIndex.from_tuples([
        (scorer, keypoint, coord)
        for keypoint in keypoint_names for coord in ("x", "y", "likelihood")
    ], names=["scorer", "bodyparts", "coords"])
    frame_count = 0
    if labels.videos:
        try:
            frame_count = int(labels.videos[0].shape[0])
        except (AttributeError, TypeError, ValueError):
            pass
    if labels.labeled_frames:
        frame_count = max(frame_count, max(lf.frame_idx for lf in labels) + 1)
    frame = pd.DataFrame(np.nan, index=range(frame_count), columns=columns)
    for labeled_frame in labels:
        if not labeled_frame.instances:
            continue
        instance = labeled_frame.instances[0]
        points = instance.numpy()
        scores = instance.points["score"] if "score" in instance.points.dtype.names \
            else np.ones(len(points))
        for index, keypoint in enumerate(keypoint_names):
            frame.loc[labeled_frame.frame_idx, (scorer, keypoint, "x")] = points[index, 0]
            frame.loc[labeled_frame.frame_idx, (scorer, keypoint, "y")] = points[index, 1]
            frame.loc[labeled_frame.frame_idx, (scorer, keypoint, "likelihood")] = scores[index]
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_hdf(h5_path, key="df_with_missing", format="table", mode="w")
    return h5_path


def _sleap_track_worker(project_path: str, videos: List[tuple[str, str]], output_dir: str,
                        gpu_id: str, batch_size: int, checkpoint: str,
                        scorer: str, peak_threshold: float) -> int:
    """Track one balanced camera subset in an isolated SLEAP CUDA process."""
    isolate_worker_output()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    checkpoint_path = Path(checkpoint)
    model_dir = checkpoint_path.parent
    for video_value, progress_value in videos:
        video = Path(video_value)
        progress = Path(progress_value)
        progress.write_text(json.dumps({"completed": 0, "total": 1}))
        output_slp = Path(output_dir) / f"{video.stem}.predictions.slp"
        command = [
            sys.executable, "-m", "sleap_nn.cli", "track",
            "--data_path", str(video), "--model_paths", str(model_dir),
            "--output_path", str(output_slp), "--device", "cuda:0",
            "--batch_size", str(batch_size),
            "--backbone_ckpt_path", str(checkpoint_path),
            "--peak_threshold", str(peak_threshold),
        ]
        subprocess.run(command, check=True)
        _sleap_predictions_to_h5(
            output_slp, Path(output_dir) / f"{video.stem}.h5", scorer
        )
        progress.write_text(json.dumps({"completed": 1, "total": 1}))
    return len(videos)


class SLEAPBackend(Pose2dBackend):
    """Cheese3D adapter for SLEAP-NN's single-instance pipeline."""

    def __init__(self, name: str, root_dir: Path, videos: List[Path],
                 keypoints: List[KeypointConfig],
                 crops: Optional[List[BoundingBox]] = None,
                 skeleton: Optional[List[List[str]]] = None,
                 source_project_path: Optional[str | Path] = None,
                 source_format: Optional[str] = None,
                 scorer: Optional[str] = None,
                 canonical_config_path: Optional[str | Path] = None,
                 **_options):
        """Initialize SLEAP with a root-level canonical network configuration."""
        super().__init__()
        self.name = name
        self.root_dir = Path(root_dir).absolute()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.videos = videos
        self.keypoints = keypoints
        self.crops = crops
        self.skeleton = skeleton or []
        self.scorer = scorer or f"SLEAP_{name}"
        # SLEAP-NN receives the backend-local path on its CLI; synchronize that
        # required working copy with the discoverable project-root YAML.
        self.canonical_config_path = (Path(canonical_config_path).absolute()
                                      if canonical_config_path else None)
        self.source_project_path = Path(source_project_path).resolve() \
            if source_project_path else None
        self.source_format = source_format
        self._selected_checkpoint: Optional[Path] = None
        if not self.config_path.is_file():
            create_sleap_training_config(
                self.project_path, self.name,
                [keypoint.label for keypoint in self.keypoints],
            )
        self._pull_canonical_config(self.config_path)
        self._push_canonical_config(self.config_path)
        if not self.labels_path.is_file():
            records = self._source_records()
            create_sleap_labels(
                records, self.labels_path,
                [keypoint.label for keypoint in self.keypoints], self.skeleton,
            )

    @property
    def project_path(self) -> Path:
        """Return the isolated SLEAP backend folder."""
        return self.root_dir

    @property
    def config_path(self) -> Path:
        """Return the SLEAP-NN YAML configuration path."""
        return self.project_path / "config.yaml"

    @property
    def labels_path(self) -> Path:
        """Return the portable SLEAP label package path."""
        return self.project_path / "labels.slp"

    @classmethod
    def from_existing(cls, project_path: Path, root_dir: Path, *args, **kwargs):
        """Importing arbitrary SLEAP projects is deferred until format stabilization."""
        raise NotImplementedError("Importing an existing SLEAP model is not implemented yet.")

    def _source_records(self) -> dict:
        """Read the optional foreign (DLC/Lightning Pose/SLEAP) seed source."""
        if self.source_project_path is None:
            return {}
        names = [keypoint.label for keypoint in self.keypoints]
        return read_foreign_records(self.source_format, self.source_project_path, names)

    def import_c3d_labels(self, videos: Dict[str, Path]):
        """Merge optional foreign seed labels with current Cheese3D annotations."""
        names = [keypoint.label for keypoint in self.keypoints]
        records = self._source_records()
        records.update(_annotation_records(videos, names))
        count = create_sleap_labels(records, self.labels_path, names, self.skeleton)
        print(f"SLEAP label package contains {count} single-instance images.")

    def export_c3d_labels(self, videos: Dict[str, Path]):
        """Export SLEAP's labels.slp back into Cheese3D annotation folders."""
        import sleap_io as sio

        if not self.labels_path.is_file():
            return
        labels = sio.load_slp(str(self.labels_path), open_videos=False)
        keypoint_names = [node.name for node in labels.skeletons[0].nodes]
        # `create_sleap_labels` groups images by their source folder name, one
        # sio.Video per folder; recover that same grouping here so labels round
        # trip back to the Cheese3D annotation folder they came from, regardless
        # of whether they originated from a DLC seed or Cheese3D's own labeler.
        by_folder: Dict[str, List[tuple]] = {}
        for labeled_frame in labels:
            if not labeled_frame.instances:
                continue
            image_path = Path(labeled_frame.video.filename[labeled_frame.frame_idx])
            by_folder.setdefault(image_path.parent.name, []).append(
                (image_path, labeled_frame.instances[0])
            )

        for name, label_path in videos.items():
            rows = by_folder.get(name)
            if not rows:
                continue
            label_path = Path(label_path)
            label_path.mkdir(parents=True, exist_ok=True)
            annotations = {keypoint: {} for keypoint in keypoint_names}
            for image_path, instance in rows:
                image_name = image_path.name
                destination_image = label_path / image_name
                if image_path.is_file() and not destination_image.exists():
                    shutil.copy2(image_path, destination_image)
                points = instance.numpy()
                for index, keypoint in enumerate(keypoint_names):
                    x, y = points[index]
                    annotations[keypoint][image_name] = [[
                        None if np.isnan(x) else float(x),
                        None if np.isnan(y) else float(y),
                    ]]
            with (label_path / "annotations.yaml").open("w") as stream:
                yaml.safe_dump(annotations, stream, sort_keys=True)

    def extract_frames(self, videos: Optional[List[Path]] = None):
        """Use Cheese3D's existing manual frame picker for SLEAP projects."""
        raise NotImplementedError("Use Cheese3D manual frame extraction for SLEAP.")

    def train(self, gpu, iterate_dataset: bool = True,
              training_settings: Optional[dict] = None):
        """Apply GUI settings and run SLEAP-NN with inherited terminal output."""
        from omegaconf import OmegaConf

        settings = training_settings or {}
        # Reload root edits at the last responsible moment before SLEAP-NN
        # validates and launches its native training configuration.
        self._pull_canonical_config(self.config_path)
        cfg = OmegaConf.load(self.config_path)
        backbone = str(settings.get("backbone", "unet_medium_rf"))
        _set_sleap_backbone(cfg, backbone)
        cfg.data_config.validation_fraction = float(
            settings.get("validation_fraction_percent", 10)
        ) / 100.0
        cfg.data_config.use_augmentations_train = bool(
            settings.get("use_augmentation", True)
        )
        geometric = cfg.data_config.augmentation_config.geometric
        geometric.rotation_min = float(settings.get("rotation_min", -15))
        geometric.rotation_max = float(settings.get("rotation_max", 15))
        geometric.scale_min = float(settings.get("scale_min", 0.9))
        geometric.scale_max = float(settings.get("scale_max", 1.1))
        geometric.translate_width = float(settings.get("translate", 0.0))
        geometric.translate_height = float(settings.get("translate", 0.0))
        trainer = cfg.trainer_config
        trainer.max_epochs = int(settings.get("epochs", trainer.max_epochs))
        trainer.train_data_loader.batch_size = int(settings.get("batch_size", 4))
        trainer.val_data_loader.batch_size = int(settings.get("val_batch_size", 4))
        trainer.optimizer.lr = float(settings.get("learning_rate", 1e-4))
        trainer.optimizer_name = str(settings.get("optimizer", "Adam"))
        # SLEAP-NN computes steps per epoch as
        #   max(natural batches in dataset, min_train_steps_per_epoch)
        # with a default minimum of 200. On a Cheese3D-sized project that
        # silently repeats the dataset many times inside one nominal "epoch"
        # -- measured here: 860 images at an effective batch of 64 gives ~14
        # natural batches, so the default 200 inflated every epoch ~15x and
        # turned a 100-epoch run into 7.5 hours. Default to a single natural
        # pass so an epoch means the same thing it does for the DLC and
        # Lightning Pose backends; a caller can still request a floor.
        trainer.min_train_steps_per_epoch = int(settings.get("min_steps_per_epoch", 0))
        exact_steps = int(settings.get("steps_per_epoch", 0))
        trainer.train_steps_per_epoch = exact_steps or None
        trainer.early_stopping.stop_training_on_plateau = bool(
            settings.get("early_stopping", True)
        )
        trainer.early_stopping.patience = int(settings.get("early_stop_patience", 10))
        trainer.model_ckpt.save_top_k = int(settings.get("save_top_k", 3))
        trainer.model_ckpt.save_last = bool(settings.get("save_last", True))
        trainer.eval.enabled = True
        trainer.eval.frequency = int(settings.get("validate_every_n_epochs", 1))
        gpu_ids = [int(value.strip()) for value in str(gpu).split(",") if value.strip()]
        trainer.trainer_accelerator = "gpu" if gpu_ids else "cpu"
        trainer.trainer_devices = len(gpu_ids) if gpu_ids else 1
        trainer.trainer_device_indices = gpu_ids or None
        trainer.trainer_strategy = "ddp" if len(gpu_ids) > 1 else "auto"
        # Preprocessing input scale is the primary memory control for
        # Cheese3D's whole-frame camera views: activation memory follows image
        # area, so 0.5 quarters it. SLEAP-NN rescales the labels with the
        # images and restores original-image coordinates at inference, so
        # predictions still come back in full-frame pixels.
        scale = float(settings.get("input_scale", 1.0))
        if not 0.0 < scale <= 1.0:
            raise ValueError("SLEAP input scale must be greater than 0 and at most 1")
        cfg.data_config.preprocessing.scale = scale
        OmegaConf.save(cfg, self.config_path)
        self._push_canonical_config(self.config_path)
        self._print_training_summary(cfg, settings, backbone, gpu_ids, scale)
        command = [sys.executable, "-m", "sleap_nn.cli", "train", str(self.config_path)]
        subprocess.run(command, cwd=self.project_path, check=True)
        self._push_canonical_config(self.config_path)

    def _print_training_summary(self, cfg, settings: dict, backbone: str,
                                gpu_ids: List[int], scale: float) -> None:
        """State the settings that actually drive SLEAP memory and runtime.

        Batch size is per GPU under DDP and an "epoch" is a step count rather
        than a dataset pass, so the numbers a user types are not the numbers
        that determine whether training fits in memory or how long it runs.
        Print both forms before launching rather than leaving them implicit.
        """
        import sleap_io as sio

        devices = max(len(gpu_ids), 1)
        per_gpu = int(cfg.trainer_config.train_data_loader.batch_size)
        effective = per_gpu * devices
        head_stride = cfg.model_config.head_configs.single_instance.confmaps.output_stride

        try:
            labels = sio.load_slp(str(self.project_path / "labels.slp"), open_videos=False)
            total = len(labels)
            train_count = int(round(total * (1.0 - float(cfg.data_config.validation_fraction))))
            height, width = labels.videos[0].shape[1:3]
            source = f"{width}x{height}"
            scaled = f"{int(width * scale)}x{int(height * scale)}"
        except Exception:
            train_count = 0
            source = scaled = "unknown"

        natural = max(1, train_count // effective) if train_count else 0
        floor = int(cfg.trainer_config.min_train_steps_per_epoch or 0)
        steps = max(natural, floor) if natural else floor

        # flush=True: this summary describes a run that is about to start, but
        # Python block-buffers stdout when it is a file or pipe while the
        # training subprocess writes to the same descriptor unbuffered. Without
        # an explicit flush the summary is emitted when the parent exits, i.e.
        # printed *after* the training output it is supposed to introduce.
        print(f"SLEAP input: {source} RGB -> {scaled} RGB (scale {scale})", flush=True)
        print(f"SLEAP batch: {per_gpu}/GPU x {devices} GPU(s) = {effective} effective",
              flush=True)
        if natural:
            note = f"minimum override {floor}" if floor else "no minimum override"
            print(f"SLEAP epoch: {natural} natural batches, {steps} steps ({note})",
                  flush=True)
        print(f"SLEAP model: {backbone}, confidence-map stride {head_stride}", flush=True)
        print(f"SLEAP training settings: {settings}", flush=True)

    def list_checkpoints(self) -> List[dict]:
        """Return SLEAP checkpoints with callback validation metrics when present."""
        import torch

        records = []
        for path in sorted((self.project_path / "models").rglob("*.ckpt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            metrics = {}
            for callback in payload.get("callbacks", {}).values():
                monitor = callback.get("monitor")
                score = callback.get("current_score", callback.get("best_model_score"))
                if monitor and score is not None:
                    metrics[str(monitor)] = float(score)
            records.append({
                "path": str(path), "epoch": payload.get("epoch"), "metrics": metrics,
            })
        return records

    def select_checkpoint(self, checkpoint: str | Path) -> None:
        """Select a SLEAP checkpoint for subsequent camera inference."""
        selected = Path(checkpoint).resolve()
        if selected not in [Path(item["path"]).resolve() for item in self.list_checkpoints()]:
            raise FileNotFoundError(f"SLEAP checkpoint is unavailable: {checkpoint}")
        self._selected_checkpoint = selected

    def selected_result_identifiers(self) -> List[str]:
        """Identify triangulation outputs belonging to selected SLEAP weights."""
        return [self._selected_checkpoint.stem] if self._selected_checkpoint else []

    def track(self, videos: Dict[str, Path], output_dir: Path,
              calibration_path: Optional[Path] = None,
              tracking_settings: Optional[dict] = None) -> bool:
        """Distribute independent camera videos across selected SLEAP GPUs."""
        settings = tracking_settings or {}
        checkpoint = self._selected_checkpoint
        if checkpoint is None:
            records = self.list_checkpoints()
            if not records:
                raise RuntimeError("Train SLEAP or select an existing checkpoint first.")
            checkpoint = Path(records[-1]["path"])
        gpu_ids = [value.strip() for value in str(settings.get("gpu_ids", "0")).split(",")
                   if value.strip()]
        batch_size = int(settings.get("batch_size", 8))
        peak_threshold = float(settings.get("peak_threshold", 0.2))
        if not 0.0 <= peak_threshold <= 1.0:
            raise ValueError("SLEAP peak threshold must be between 0 and 1")
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_files = {
            name: Path(output_dir) / f".{Path(video).stem}.sleap-progress.json"
            for name, video in videos.items()
        }
        assignments = partition_videos_by_gpu(list(videos.values()), gpu_ids)
        name_by_video = {Path(video): name for name, video in videos.items()}
        context = __import__("multiprocessing").get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=len(assignments), mp_context=context)
        futures = []
        for gpu_id, assigned in assignments:
            worker_videos = [
                (str(video), str(progress_files[name_by_video[Path(video)]]))
                for video in assigned
            ]
            futures.append(pool.submit(
                _sleap_track_worker, str(self.project_path), worker_videos,
                str(output_dir), gpu_id, batch_size, str(checkpoint), self.scorer,
                peak_threshold,
            ))
        try:
            monitor_camera_progress(futures, progress_files, unit="video")
        finally:
            shutdown_completed_process_pool(pool)
        return True


register_pose_backend("sleap", SLEAPBackend)
