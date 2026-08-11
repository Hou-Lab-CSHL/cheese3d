import json
import os
import shutil
import subprocess
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from cheese3d.backends.core import (Pose2dBackend, register_pose_backend,
                                    partition_videos_by_gpu, monitor_camera_progress)
from cheese3d.config import KeypointConfig
from cheese3d.utils import BoundingBox


def _dlc_image_path(project_path: Path, index_value) -> Path:
    """Resolve one DLC label-table index into its source image path."""
    if isinstance(index_value, tuple):
        relative_parts = [str(part) for part in index_value]
    else:
        # DLC CSV/H5 indexes use POSIX separators even when read on another OS.
        relative_parts = list(Path(str(index_value)).parts)
    return project_path.joinpath(*relative_parts)


def _read_dlc_label_table(label_dir: Path) -> Optional[pd.DataFrame]:
    """Read the preferred H5 label table, falling back to its DLC CSV peer."""
    h5_files = sorted(label_dir.glob("CollectedData_*.h5"))
    if h5_files:
        return pd.read_hdf(h5_files[0])
    csv_files = sorted(label_dir.glob("CollectedData_*.csv"))
    if csv_files:
        return pd.read_csv(csv_files[0], header=[0, 1, 2], index_col=[0, 1, 2])
    return None


def convert_dlc_labels_to_lightning_pose(
    dlc_project_path: str | Path,
    lightning_pose_project_path: str | Path,
    keypoint_names: Optional[Sequence[str]] = None,
    copy_images: bool = True,
) -> dict:
    """Convert a DLC project's labeled frames into a portable LP dataset.

    Lightning Pose consumes DLC-style three-row CSV headers directly, but image
    paths must resolve beneath one data root. This converter merges every DLC
    ``labeled-data/*/CollectedData_*`` table, normalizes all scorer names, and
    optionally copies the referenced images so the LP project remains usable if
    the source DLC project is moved.

    Args:
        dlc_project_path: DLC project containing ``labeled-data``.
        lightning_pose_project_path: Destination model directory.
        keypoint_names: Optional canonical keypoint order. Missing keypoints are
            represented by NaNs instead of dropping frames or entire folders.
        copy_images: Copy images into the LP data directory when true.

    Returns:
        Conversion summary containing output paths, row count, and keypoints.
    """
    dlc_project_path = Path(dlc_project_path).resolve()
    lp_project_path = Path(lightning_pose_project_path).resolve()
    labeled_data = dlc_project_path / "labeled-data"
    if not labeled_data.is_dir():
        raise FileNotFoundError(f"DLC labeled-data directory not found: {labeled_data}")

    data_dir = lp_project_path / "data"
    output_labeled_data = data_dir / "labeled-data"
    output_labeled_data.mkdir(parents=True, exist_ok=True)
    # A video directory is required by Lightning Pose's data-path validation even
    # for a fully supervised model that does not consume unlabeled video losses.
    (data_dir / "videos").mkdir(parents=True, exist_ok=True)

    tables: List[pd.DataFrame] = []
    discovered_keypoints: List[str] = []
    copied_images = 0
    missing_images: List[str] = []
    for label_dir in sorted(path for path in labeled_data.iterdir() if path.is_dir()):
        table = _read_dlc_label_table(label_dir)
        if table is None or table.empty:
            continue
        if not isinstance(table.columns, pd.MultiIndex) or table.columns.nlevels < 3:
            raise ValueError(f"DLC label table does not have a 3-level header: {label_dir}")

        for name in map(str, table.columns.get_level_values(1).unique()):
            if name not in discovered_keypoints:
                discovered_keypoints.append(name)
        destination_rows = []
        for index_value in table.index:
            source_image = _dlc_image_path(dlc_project_path, index_value)
            destination_image = output_labeled_data / label_dir.name / source_image.name
            if not source_image.is_file():
                missing_images.append(str(source_image))
                destination_rows.append(None)
                continue
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            if copy_images and (
                not destination_image.exists()
                or destination_image.stat().st_mtime < source_image.stat().st_mtime
            ):
                shutil.copy2(source_image, destination_image)
                copied_images += 1
            destination_rows.append(
                str(Path("labeled-data") / label_dir.name / source_image.name)
            )

        keep = np.array([row is not None for row in destination_rows], dtype=bool)
        table = table.loc[keep].copy()
        destination_rows = [row for row in destination_rows if row is not None]
        # Former DLC scorer names are intentionally not retained: LP only needs
        # one consistent top-level label across merged source projects.
        table.columns = pd.MultiIndex.from_tuples([
            ("lightning_pose", str(bodypart), str(coord))
            for _, bodypart, coord in table.columns
        ], names=["scorer", "bodyparts", "coords"])
        table.index = pd.Index(destination_rows, name="image")
        tables.append(table)

    if not tables:
        raise RuntimeError(f"No DLC CollectedData label tables found under {labeled_data}")
    if missing_images:
        sample = "\n".join(f"  - {path}" for path in missing_images[:10])
        raise FileNotFoundError(
            f"{len(missing_images)} DLC label images are missing; conversion stopped.\n{sample}"
        )

    ordered_keypoints = list(keypoint_names or discovered_keypoints)
    columns = pd.MultiIndex.from_tuples([
        ("lightning_pose", name, coord)
        for name in ordered_keypoints
        for coord in ("x", "y")
    ], names=["scorer", "bodyparts", "coords"])
    normalized_tables = [table.reindex(columns=columns) for table in tables]
    merged = pd.concat(normalized_tables, axis=0)
    # Keep the newest occurrence when repeated imports contain the same image.
    merged = merged.loc[~merged.index.duplicated(keep="last")].sort_index()
    # Lightning Pose expects exactly three CSV header rows. A named index writes
    # a fourth `image,,,,` row that its dataset mistakenly treats as an image.
    merged.index.name = None
    output_csv = data_dir / "CollectedData.csv"
    merged.to_csv(output_csv)

    return {
        "data_dir": data_dir,
        "csv_file": output_csv,
        "num_frames": len(merged),
        "num_keypoints": len(ordered_keypoints),
        "keypoint_names": ordered_keypoints,
        "copied_images": copied_images,
    }


def create_lightning_pose_training_config(
    project_path: str | Path,
    conversion: dict,
    model_name: str,
    image_height: int = 512,
    image_width: int = 640,
    overrides: Optional[dict] = None,
) -> Path:
    """Write a supervised Lightning Pose 2.2 training configuration.

    The defaults mirror the official single-view template but disable automatic
    post-training video inference because Cheese3D performs inference separately.
    """
    from omegaconf import OmegaConf

    project_path = Path(project_path).resolve()
    cfg = OmegaConf.create({
        "data": {
            "image_resize_dims": {"height": image_height, "width": image_width},
            "data_dir": str(conversion["data_dir"]),
            "video_dir": str(conversion["data_dir"] / "videos"),
            "csv_file": Path(conversion["csv_file"]).name,
            "num_keypoints": conversion["num_keypoints"],
            "keypoint_names": conversion["keypoint_names"],
            "downsample_factor": 2,
            "mirrored_column_matches": None,
            "columns_for_singleview_pca": None,
        },
        "training": {
            "imgaug": "dlc", "imgaug_hflip": False,
            "train_batch_size": 16, "val_batch_size": 32, "test_batch_size": 32,
            "train_prob": 0.95, "val_prob": 0.05, "train_frames": 1,
            "num_gpus": 1, "unfreezing_epoch": 20,
            "min_epochs": 300, "max_epochs": 300,
            "log_every_n_steps": 10, "check_val_every_n_epoch": 5,
            "ckpt_every_n_epochs": None, "early_stopping": False,
            "early_stop_patience": 3, "rng_seed_data_pt": 0,
            "rng_seed_model_pt": 0, "optimizer": "Adam",
            "optimizer_params": {"learning_rate": 1e-3},
            "lr_scheduler": "multisteplr",
            "lr_scheduler_params": {
                "multisteplr": {"milestones": [150, 200, 250], "gamma": 0.5}
            },
            "uniform_heatmaps_for_nan_keypoints": True,
        },
        "model": {
            "losses_to_use": [], "backbone": "resnet50_animal_ap10k",
            "model_type": "heatmap", "heatmap_loss_type": "mse",
            "model_name": model_name, "checkpoint": None,
        },
        "dali": {
            "base": {"train": {"sequence_length": 32}, "predict": {"sequence_length": 96}},
            "context": {"train": {"batch_size": 16}, "predict": {"sequence_length": 96}},
        },
        "losses": {
            "pca_multiview": {"log_weight": 11.0, "components_to_keep": 3, "epsilon": None},
            "pca_singleview": {"log_weight": 11.0, "components_to_keep": 0.99, "epsilon": None},
            "temporal": {"log_weight": 11.0, "epsilon": 20.0, "prob_threshold": 0.05},
        },
        "eval": {
            "predict_vids_after_training": False,
            "test_videos_directory": "${data.video_dir}",
            "save_vids_after_training": False, "colormap": "cool",
            "confidence_thresh_for_vid": 0.90,
        },
        "callbacks": {
            "anneal_weight": {"attr_name": "total_unsupervised_importance",
                              "init_val": 0.0, "increase_factor": 0.01,
                              "final_val": 1.0, "freeze_until_epoch": 60}
        },
    })
    if overrides:
        cfg.merge_with(overrides)
    config_path = project_path / "config.yaml"
    OmegaConf.save(cfg, config_path)
    return config_path

def is_lightning_pose_video(video: str | Path) -> bool:
    """Return whether a video has the MP4 H.264 format used by the LP app."""
    video = Path(video)
    if video.suffix.lower() != ".mp4":
        return False
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=format_name:stream=codec_name,pix_fmt",
            "-of", "json",
            str(video)
        ], capture_output=True, check=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    streams = probe.get("streams", [])
    format_names = probe.get("format", {}).get("format_name", "").split(",")

    return ("mp4" in format_names and len(streams) == 1 and
            streams[0].get("codec_name") == "h264" and
            streams[0].get("pix_fmt") == "yuv420p")

def preprocess_lightning_pose_video(video: str | Path, output_dir: str | Path) -> Path:
    """Convert a video to the MP4 H.264 format used by the Lightning Pose app."""
    video = Path(video)
    if is_lightning_pose_video(video):
        return video
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video.stem}.mp4"
    if (output_path.exists() and
        output_path.stat().st_mtime >= video.stat().st_mtime and
        is_lightning_pose_video(output_path)):
        return output_path
    try:
        subprocess.run([
            "ffmpeg",
            "-i", str(video),
            "-loglevel", "info", "-stats",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-g", "1",
            "-preset", "medium",
            "-crf", "23",
            "-vf", "setsar=1",
            "-an",
            "-y", str(output_path)
        ], check=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is required to preprocess Lightning Pose videos.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Failed to preprocess video {video} for Lightning Pose.") from error

    return output_path

def read_lp_preds(csv_path: str | Path, scorer: Optional[str] = None) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    keep_cols = []
    for col in df.columns:
        top, bodypart, coord = col
        if str(top).lower() == "set":
            continue
        if pd.isna(bodypart) or pd.isna(coord):
            continue
        if str(coord) not in {"x", "y", "likelihood"}:
            continue
        keep_cols.append(col)
    df = df.loc[:, keep_cols]
    if scorer is not None:
        df.columns = pd.MultiIndex.from_tuples(
            [(scorer, bodypart, coord) for _, bodypart, coord in df.columns],
            names=["scorer", "bodyparts", "coords"]
        )
    else:
        df.columns = pd.MultiIndex.from_tuples(list(df.columns),
                                               names=["scorer", "bodyparts", "coords"])

    return df.infer_objects()

def dlc_df_to_h5(df: pd.DataFrame, h5_path: str | Path, scorer: Optional[str] = None,
                 overwrite: bool = False) -> Path:
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = [col for col in df.columns if col[2] in {"x", "y", "likelihood"}]
    df = df.loc[:, keep_cols].copy()
    if scorer is not None:
        df.columns = pd.MultiIndex.from_tuples(
            [(scorer, bodypart, coord) for _, bodypart, coord in df.columns],
            names=["scorer", "bodyparts", "coords"]
        )
    if not h5_path.exists() or overwrite:
        df.to_hdf(h5_path, key="df_with_missing", format="table", mode="w")

    return h5_path

def lp_csv_to_dlc_h5(csv_path: str | Path, h5_path: str | Path,
                     scorer: Optional[str] = None, overwrite: bool = False) -> Path:
    return dlc_df_to_h5(read_lp_preds(csv_path, scorer=scorer), h5_path,
                        scorer=scorer, overwrite=overwrite)


def _lp_track_gpu_worker(project_path: str, videos: List[tuple[str, str]], output_dir: str,
                         gpu_id: str, batch_size: int, checkpoint: Optional[str],
                         scorer: Optional[str]) -> int:
    """Analyze one camera subset in an isolated Lightning Pose GPU process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    from lightning_pose.api import Model
    from lightning_pose.api.model import load_model_from_checkpoint

    model = Model.from_dir(project_path)
    model.cfg.training.test_batch_size = batch_size
    if checkpoint:
        model.model = load_model_from_checkpoint(
            cfg=model.cfg, ckpt_file=checkpoint, eval=True, skip_data_module=True
        )
    print(f"Lightning Pose GPU {gpu_id} analyzing {len(videos)} video(s)",
          flush=True)
    for video_value, progress_value in videos:
        video = Path(video_value)
        preprocessed = preprocess_lightning_pose_video(
            video, Path(project_path) / "preprocessed-videos"
        )
        model.predict_on_video_file(
            preprocessed, compute_metrics=False, generate_labeled_video=False,
            progress_file=Path(progress_value),
        )
        prediction_csv = model.video_preds_dir() / f"{video.stem}.csv"
        lp_csv_to_dlc_h5(
            prediction_csv, Path(output_dir) / f"{video.stem}.h5",
            scorer=scorer, overwrite=checkpoint is not None,
        )
    return len(videos)

class LightningPoseBackend(Pose2dBackend):
    def __init__(self,
                 name: str,
                 root_dir: Path,
                 videos: List[Path],
                 keypoints: List[KeypointConfig],
                 crops: Optional[List[BoundingBox]] = None,
                 scorer: Optional[str] = None,
                 dlc_project_path: Optional[str | Path] = None,
                 copy_dlc_images: bool = True,
                 image_height: int = 512,
                 image_width: int = 640,
                 **cfg_options):
        from lightning_pose.api import Model

        super().__init__()
        self.name = name
        self.root_dir = Path(root_dir).absolute()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.videos = videos
        self.keypoints = keypoints
        self.crops = crops
        self.scorer = scorer
        self._selected_checkpoint: Optional[Path] = None
        if not (self.project_path / "config.yaml").exists():
            if dlc_project_path is None:
                raise FileNotFoundError(
                    f"Lightning Pose config not found at {self.project_path / 'config.yaml'}. "
                    "Set model.backend_options.dlc_project_path to convert a DLC project."
                )
            # DLC coordinates already use Lightning Pose's accepted CSV format;
            # conversion mainly makes paths portable and merges label folders.
            conversion = convert_dlc_labels_to_lightning_pose(
                dlc_project_path=dlc_project_path,
                lightning_pose_project_path=self.project_path,
                keypoint_names=[keypoint.label for keypoint in self.keypoints],
                copy_images=copy_dlc_images,
            )
            create_lightning_pose_training_config(
                project_path=self.project_path,
                conversion=conversion,
                model_name=self.name,
                image_height=image_height,
                image_width=image_width,
            )
            print(
                f"Converted {conversion['num_frames']} DLC labeled frames and "
                f"{conversion['num_keypoints']} keypoints into {self.project_path}"
            )
        self._update_config(**cfg_options)
        self.model = Model.from_dir(self.project_path)

    @classmethod
    def from_existing(cls,
                      project_path: Path,
                      root_dir: Path,
                      *args,
                      **kwargs):
        raise NotImplementedError("Importing existing Lightning Pose models is not implemented.")

    def _update_config(self, **cfg_options):
        from lightning_pose.api import ModelConfig
        from omegaconf import OmegaConf

        cfg = ModelConfig.from_yaml_file(self.project_path / "config.yaml")
        cfg.cfg.merge_with(cfg_options)
        OmegaConf.save(cfg.cfg, self.project_path / "config.yaml")

    @property
    def project_path(self):
        return self.root_dir

    def import_c3d_labels(self, videos: Dict[str, Path]):
        """Merge Cheese3D ``annotations.yaml`` labels into the LP training CSV."""
        csv_path = self.project_path / "data" / "CollectedData.csv"
        if csv_path.exists():
            existing = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        else:
            existing = pd.DataFrame()
        columns = pd.MultiIndex.from_tuples([
            ("lightning_pose", keypoint.label, coord)
            for keypoint in self.keypoints
            for coord in ("x", "y")
        ], names=["scorer", "bodyparts", "coords"])

        updates: List[pd.DataFrame] = []
        for name, label_path in videos.items():
            label_path = Path(label_path)
            annotations_path = label_path / "annotations.yaml"
            if not annotations_path.is_file():
                continue
            with annotations_path.open("r") as stream:
                annotations = yaml.safe_load(stream) or {}
            image_names = sorted({
                image_name
                for keypoint_labels in annotations.values()
                for image_name in (keypoint_labels or {}).keys()
            })
            if not image_names:
                continue
            frame = pd.DataFrame(index=pd.Index([
                str(Path("labeled-data") / name / image_name)
                for image_name in image_names
            ], name="image"), columns=columns, dtype=float)
            for keypoint in self.keypoints:
                for image_name, points in (annotations.get(keypoint.label) or {}).items():
                    if not points or not points[0]:
                        continue
                    x, y = points[0]
                    row_name = str(Path("labeled-data") / name / image_name)
                    frame.loc[row_name, ("lightning_pose", keypoint.label, "x")] = x
                    frame.loc[row_name, ("lightning_pose", keypoint.label, "y")] = y

            output_dir = self.project_path / "data" / "labeled-data" / name
            output_dir.mkdir(parents=True, exist_ok=True)
            for image_name in image_names:
                source = label_path / image_name
                destination = output_dir / image_name
                if source.is_file() and (
                    not destination.exists() or source.resolve() != destination.resolve()
                ):
                    shutil.copy2(source, destination)
            updates.append(frame)

        if not updates:
            return
        # Former LP behavior rejected Cheese3D annotations entirely. New rows
        # replace matching DLC imports while all unrelated training rows remain.
        combined = pd.concat([existing.reindex(columns=columns), *updates])
        combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
        # Keep LP's three-row DLC-style header when annotations are appended;
        # an index label would otherwise become a nonexistent `data/image` row.
        combined.index.name = None
        combined.to_csv(csv_path)
        self._synchronize_config_keypoints(combined)

    def export_c3d_labels(self, videos: Dict[str, Path]):
        """Export LP training labels back into Cheese3D annotation folders."""
        csv_path = self.project_path / "data" / "CollectedData.csv"
        if not csv_path.is_file():
            return
        labels = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        for name, label_path in videos.items():
            label_path = Path(label_path)
            prefix = f"labeled-data/{name}/"
            rows = labels.loc[[str(index).startswith(prefix) for index in labels.index]]
            if rows.empty:
                continue
            label_path.mkdir(parents=True, exist_ok=True)
            annotations = {keypoint.label: {} for keypoint in self.keypoints}
            for image_index, row in rows.iterrows():
                image_name = Path(str(image_index)).name
                source_image = self.project_path / "data" / str(image_index)
                destination_image = label_path / image_name
                if source_image.is_file() and not destination_image.exists():
                    shutil.copy2(source_image, destination_image)
                for keypoint in self.keypoints:
                    try:
                        x = row[("lightning_pose", keypoint.label, "x")]
                        y = row[("lightning_pose", keypoint.label, "y")]
                    except KeyError:
                        x = y = np.nan
                    annotations[keypoint.label][image_name] = [[
                        None if pd.isna(x) else float(x),
                        None if pd.isna(y) else float(y),
                    ]]
            with (label_path / "annotations.yaml").open("w") as stream:
                yaml.safe_dump(annotations, stream, sort_keys=True)

    def _synchronize_config_keypoints(self, labels: pd.DataFrame) -> None:
        """Keep LP config metadata consistent with the converted label columns."""
        from omegaconf import OmegaConf

        config_path = self.project_path / "config.yaml"
        cfg = OmegaConf.load(config_path)
        names = list(dict.fromkeys(map(str, labels.columns.get_level_values(1))))
        cfg.data.keypoint_names = names
        cfg.data.num_keypoints = len(names)
        OmegaConf.save(cfg, config_path)

    def extract_frames(self, videos: Optional[List[Path]] = None):
        raise NotImplementedError("Use Lightning Pose tooling to extract frames.")

    def train(self, gpu, iterate_dataset: bool = True,
              training_settings: Optional[dict] = None):
        """Train the converted labels and refresh the inference model handle.

        ``iterate_dataset`` is retained for API compatibility with DLC. Lightning
        Pose creates a fresh checkpoint series in the same model directory and
        does not use DLC's numbered dataset-iteration folders.
        """
        from lightning_pose.train import train as train_lightning_pose
        from lightning_pose.api import Model, ModelConfig

        cfg = ModelConfig.from_yaml_file(self.project_path / "config.yaml")
        settings = training_settings or {}
        cfg.cfg.training.max_epochs = settings.get("epochs", cfg.cfg.training.max_epochs)
        cfg.cfg.training.min_epochs = min(
            settings.get("epochs", cfg.cfg.training.min_epochs),
            cfg.cfg.training.max_epochs,
        )
        cfg.cfg.training.train_batch_size = settings.get(
            "batch_size", cfg.cfg.training.train_batch_size
        )
        cfg.cfg.training.optimizer_params.learning_rate = settings.get(
            "learning_rate", cfg.cfg.training.optimizer_params.learning_rate
        )
        cfg.cfg.training.imgaug = settings.get("imgaug", cfg.cfg.training.imgaug)
        cfg.cfg.training.imgaug_hflip = settings.get(
            "horizontal_flip", cfg.cfg.training.imgaug_hflip
        )
        cfg.cfg.training.train_prob = settings.get("train_prob", cfg.cfg.training.train_prob)
        cfg.cfg.training.val_prob = settings.get("val_prob", cfg.cfg.training.val_prob)
        cfg.cfg.training.unfreezing_epoch = settings.get(
            "unfreezing_epoch", cfg.cfg.training.unfreezing_epoch
        )
        cfg.cfg.training.early_stopping = settings.get(
            "early_stopping", cfg.cfg.training.early_stopping
        )
        cfg.cfg.training.early_stop_patience = settings.get(
            "early_stop_patience", cfg.cfg.training.early_stop_patience
        )
        cfg.cfg.training.ckpt_every_n_epochs = settings.get(
            "save_every_n_epochs", cfg.cfg.training.ckpt_every_n_epochs
        )
        cfg.cfg.training.check_val_every_n_epoch = settings.get(
            "validate_every_n_epochs", cfg.cfg.training.check_val_every_n_epoch
        )
        gpu_ids = [item.strip() for item in str(gpu).split(",") if item.strip()]
        cfg.cfg.training.num_gpus = len(gpu_ids)
        cfg.validate()
        # Former behavior stopped here with NotImplementedError; training now uses
        # LP's public train function and writes checkpoints beside config.yaml.
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        print(
            f"Training Lightning Pose model '{self.name}' on GPU {gpu}; "
            f"data={cfg.cfg.data.csv_file}"
        )
        print(f"Lightning Pose training settings: {settings}")
        self.model = train_lightning_pose(
            cfg.cfg,
            model_dir=self.project_path,
            skip_evaluation=False,
        )
        # Re-open through the stable inference API so future tracking calls find
        # the checkpoint exactly as they would after restarting Cheese3D.
        self.model = Model.from_dir(self.project_path)

    def list_checkpoints(self) -> List[dict]:
        """List LP checkpoints with their monitored validation score."""
        import torch
        records = []
        for path in self.project_path.rglob("*.ckpt"):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            metrics = {}
            for callback in payload.get("callbacks", {}).values():
                monitor = callback.get("monitor")
                score = callback.get("current_score", callback.get("best_model_score"))
                if monitor and score is not None:
                    metrics[str(monitor)] = float(score)
            records.append({"path": str(path), "epoch": payload.get("epoch"),
                            "metrics": metrics})
        records.sort(key=lambda item: (item["epoch"] is None, item["epoch"] or -1))
        return records

    def select_checkpoint(self, checkpoint: str | Path) -> None:
        """Remember the exact LP checkpoint to load when tracking starts."""
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Lightning Pose checkpoint is unavailable: {path}")
        self._selected_checkpoint = path

    def track(self,
              videos: Dict[str, Path],
              output_dir: Path,
              calibration_path: Optional[Path] = None,
              tracking_settings: Optional[dict] = None) -> bool:
        """Track with explicit LP inference device and batch configuration."""
        from lightning_pose.utils.predictions import predict_video
        from lightning_pose.api.model import load_model_from_checkpoint

        if self.model is None:
            raise RuntimeError("Lightning Pose model is not initialized; create or train it first.")
        settings = tracking_settings or {}
        gpu_ids = [item.strip() for item in str(settings.get("gpu_ids", "0")).split(",")
                   if item.strip()]
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        self.model.cfg.training.test_batch_size = settings.get("batch_size", 32)
        print(f"Lightning Pose tracking settings: active_gpu="
              f"{gpu_ids[0] if gpu_ids else 'cpu'}, "
              f"batch_size={self.model.cfg.training.test_batch_size}, "
              f"requested_gpus={gpu_ids}")
        if self._selected_checkpoint is not None:
            # Loading is deferred from the Select event to the tracking worker so
            # choosing a checkpoint never freezes the interactive GUI.
            self.model.model = load_model_from_checkpoint(
                cfg=self.model.cfg, ckpt_file=str(self._selected_checkpoint),
                eval=True, skip_data_module=True,
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        # resolve paths to absolute
        videos = {view: Path(video).resolve() for view, video in videos.items()}
        # list of videos that we need to generate
        # exit early if there are none
        missing_videos = [video for video in videos.values()
                          if self._selected_checkpoint is not None
                          or not (output_dir / f"{video.stem}.h5").exists()]
        if len(missing_videos) == 0:
            return True
        if len(gpu_ids) > 1 and len(missing_videos) > 1:
            assignments = partition_videos_by_gpu(missing_videos, gpu_ids)
            print(f"Lightning Pose multi-GPU video assignments: {assignments}")
            context = multiprocessing.get_context("spawn")
            with tempfile.TemporaryDirectory(prefix="cheese3d-lp-progress-") as progress_dir:
                progress_files = {
                    video.stem: Path(progress_dir) / f"{video.stem}.json"
                    for video in missing_videos
                }
                with ProcessPoolExecutor(max_workers=len(assignments), mp_context=context) as pool:
                    futures = [pool.submit(
                        _lp_track_gpu_worker, str(self.project_path),
                        [(str(video), str(progress_files[video.stem])) for video in assigned],
                        str(output_dir), gpu_id, int(settings.get("batch_size", 32)),
                        str(self._selected_checkpoint) if self._selected_checkpoint else None,
                        self.scorer,
                    ) for gpu_id, assigned in assignments]
                    monitor_camera_progress(futures, progress_files)
            return True
        # list of videos that lp model has already created internal preds for
        existing = [self.model.video_preds_dir() / f"{video.stem}.csv"
                    for video in missing_videos]
        if not all(path.exists() for path in existing):
            # NOTE: currently the config is saying multiview
            #       but lenny says this is a SVT
            # if self.model.config.is_multi_view():
            #     view_names = list(self.model.config.cfg.data.view_names)
            #     if set(view_names) != set(videos.keys()):
            #         raise RuntimeError("Session view names do not match view names for trained LP model!"
            #                            f" Expected {set(view_names)}, got {set(videos.keys())}")
            #     # get input/output file path ordering matching view order
            #     ordered_videos = [videos[view_name] for view_name in view_names]
            #     output_files = [self.model.video_preds_dir() / f"{video.stem}.csv"
            #                     for video in ordered_videos]
            #     existing_outputs = [path for path in output_files if path.exists()]
            #     if len(existing_outputs) > 0:
            #         raise RuntimeError("Lightning Pose multiview tracking would overwrite existing "
            #                            f"prediction files: {existing_outputs}")
            #     # automated collection utility does not match view names for C3D format correctly
            #     # do this manually (otherwise could use model.predict_on_video_file_multiview)
            #     self.model._load()
            #     predict_video(video_file=[str(video) for video in ordered_videos],
            #                   model=self.model,
            #                   output_pred_file=[str(path) for path in output_files])
            # else:
                for video in missing_videos:
                    print(f"single view tracking for {video}")
                    prediction_csv = self.model.video_preds_dir() / f"{video.stem}.csv"
                    if prediction_csv.exists() and self._selected_checkpoint is None:
                        continue
                    preprocessed_video = preprocess_lightning_pose_video(
                        video, self.project_path / "preprocessed-videos"
                    )
                    self.model.predict_on_video_file(preprocessed_video,
                                                     compute_metrics=False,
                                                     generate_labeled_video=False)
        for video in missing_videos:
            csv_path = self.model.video_preds_dir() / f"{video.stem}.csv"
            h5_path = output_dir / f"{video.stem}.h5"
            lp_csv_to_dlc_h5(csv_path, h5_path, scorer=self.scorer,
                             overwrite=self._selected_checkpoint is not None)
            print(f"Wrote {h5_path}")

        return True

register_pose_backend("lightning_pose", LightningPoseBackend)
