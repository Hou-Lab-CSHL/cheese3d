import os
import shutil
import yaml
import pandas as pd
import multiprocessing
import json
import re
import tempfile
from contextlib import contextmanager, redirect_stderr
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from omegaconf import OmegaConf

from cheese3d.backends.core import (Pose2dBackend, register_pose_backend,
                                    isolate_worker_output, partition_videos_by_gpu,
                                    monitor_camera_progress, read_foreign_records,
                                    shutdown_completed_process_pool, write_dlc_records)
from cheese3d.config import KeypointConfig
from cheese3d.utils import maybe, reglob, BoundingBox, VideoFrames, dlc_folder_to_components


# DLC3 reports the full architecture list through
# ``pose_estimation_pytorch.available_models()``. Keeping the UI-facing subset
# here avoids importing DLC when the Lightning Pose Pixi environment opens
# Cheese3D, while train() verifies each name against the active DLC
# installation before creating or replacing a training shuffle.
#
# Bottom-up single-animal architectures only. DLC3 also ships top-down and
# conditional-top-down networks, which are deliberately excluded: they crop to
# a detected bounding box (``data.train.top_down_crop``) instead of sampling
# crops from the full frame (``data.train.crop_sampling``), and the CTD family
# additionally requires a prior pose as a conditioning input. Cheese3D tracks a
# single head-fixed animal per camera with no detector stage, so none of them
# apply here. Sweeping all 34 confirmed the split: every architecture below
# trains, while the 20 excluded ones fail on the crop or conditioning
# requirement.
DLC3_PYTORCH_MODELS = (
    "cspnext_m", "cspnext_s", "cspnext_x",
    "dekr_w18", "dekr_w32", "dekr_w48",
    "dlcrnet_stride16_ms5", "dlcrnet_stride32_ms5",
    "hrnet_w18", "hrnet_w32", "hrnet_w48",
    "resnet_101", "resnet_50",
)


def _paf_graph_from_skeleton(bodyparts: List[str],
                             skeleton: List[List[str]]) -> List[List[int]]:
    """Convert Cheese3D bodypart-name edges to DLC's integer PAF graph.

    Every edge comes out in ascending order, which DLC requires even though a
    PAF edge is undirected. Its graph pruning takes edges back from a NetworkX
    spanning tree, sorts them, and looks them up in the graph given here::

        root_edges.append(full_graph.index(sorted(edge)))   # prune_paf_graph.py

    A descending edge is therefore unfindable, and evaluation dies with
    ``ValueError: [0, 2] is not in list`` after training has already
    succeeded. Cheese3D's keypoint groups are triangles -- ``a-b``, ``b-c``,
    ``c-a`` -- so the closing edge of each one is descending as written: 8 of
    the 28 default edges. Sorting also merges an ``a-b``/``b-a`` pair, which
    is why duplicates are dropped rather than left to become two PAF channels
    for one connection.
    """
    indices = {name: index for index, name in enumerate(bodyparts)}
    invalid = [edge for edge in skeleton if len(edge) != 2 or
               edge[0] not in indices or edge[1] not in indices]
    if invalid:
        raise ValueError(f"Cheese3D skeleton contains invalid DLC edges: {invalid}")
    graph: List[List[int]] = []
    for edge in skeleton:
        pair = sorted((indices[edge[0]], indices[edge[1]]))
        if pair not in graph:
            graph.append(pair)
    if not graph:
        raise ValueError("DLCRNet requires a non-empty Cheese3D skeleton/PAF graph")
    return graph


# DLCRNet's multi-scale branches are not replicated correctly by the
# DataParallel wrapper DLC3 uses for multi-GPU training: training dies almost
# immediately with "Expected all tensors to be on the same device, but got
# weight is on cuda:0 ... and input is on cuda:1". Confirmed by direct
# comparison -- both dlcrnet variants fail within ~19 s on two GPUs and train a
# full epoch on one. Fall back to a single GPU rather than failing, since a
# slower run is a far better outcome than no run.
DLC_SINGLE_GPU_ARCHITECTURES = ("dlcrnet",)


def _supported_training_gpus(network_architecture: str, gpu_ids: List[int]) -> List[int]:
    """Restrict architectures that cannot train under DataParallel to one GPU."""
    if len(gpu_ids) > 1 and str(network_architecture).startswith(
        DLC_SINGLE_GPU_ARCHITECTURES
    ):
        print(
            f"DLC3 {network_architecture} does not support multi-GPU DataParallel; "
            f"training on GPU {gpu_ids[0]} only."
        )
        return gpu_ids[:1]
    return gpu_ids


# DLC3 gives ResNet a GroupNorm variant (``resnet50_gn``) but builds the other
# architectures with ordinary BatchNorm and frozen pretrained running
# statistics. GroupNorm does not depend on batch statistics at all, so ResNet
# tolerates a learning rate that pushes activations away from the distribution
# those frozen statistics encode -- and the BatchNorm backbones do not.
#
# Measured on hrnet_w32 with everything else identical: at 5e-4 the training
# loss sat at 0.0149 for 30+ epochs with validation RMSE 419 px and mAP 0.00,
# i.e. the model collapsed to predicting background everywhere. At 1e-4 the
# same run reached RMSE 23.04 px and mAP 87.58 in ten epochs, and 1e-5 also
# learned (final loss 0.00427 against 1e-4's 0.00308). The usable range is
# therefore at or below 1e-4 for these backbones.
DLC_DEFAULT_LEARNING_RATE = 5e-4
DLC_BATCHNORM_LEARNING_RATE = 1e-4
DLC_BATCHNORM_ARCHITECTURES = ("hrnet", "dekr", "cspnext", "dlcrnet")


def default_learning_rate(network_architecture: str) -> float:
    """Return the learning rate that suits this architecture's normalization."""
    if str(network_architecture).startswith(DLC_BATCHNORM_ARCHITECTURES):
        return DLC_BATCHNORM_LEARNING_RATE
    return DLC_DEFAULT_LEARNING_RATE


def resolve_learning_rate(network_architecture: str, settings: Dict) -> float:
    """Pick the learning rate, warning when an explicit one looks unsafe."""
    recommended = default_learning_rate(network_architecture)
    requested = settings.get("learning_rate")
    if requested is None:
        return recommended
    requested = float(requested)
    if requested > recommended and recommended == DLC_BATCHNORM_LEARNING_RATE:
        print(
            f"DLC3 warning: {network_architecture} uses BatchNorm with frozen "
            f"pretrained statistics and is unstable above {recommended:g}; "
            f"training at {requested:g} risks a flat loss that never recovers."
        )
    return requested


def _shuffle_from_created_splits(splits, default: int = 1) -> int:
    """Return DLC's actual newly-created shuffle instead of assuming shuffle 1."""
    if not isinstance(splits, (list, tuple)):
        return default
    created = []
    for split in splits:
        if isinstance(split, (list, tuple)) and len(split) >= 2:
            try:
                created.append(int(split[1]))
            except (TypeError, ValueError):
                continue
    return max(created, default=default)


def _include_compatible_labeled_data(config_path: Path) -> dict:
    """Add every DLC-compatible labeled-data folder to ``video_sets``.

    DLC selects annotation folders by taking the stem of each ``video_sets``
    key. Imported Cheese3D labels often have no corresponding source video, so
    limiting that mapping to the six active camera videos silently reduced a
    1,940-image project to 286 images. Placeholder paths are safe here: DLC's
    dataset merger uses their stems and reads the actual images from
    ``labeled-data``.
    """
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text()) or {}
    project_path = Path(config.get("project_path", config_path.parent))
    expected_bodyparts = list(config.get("bodyparts", []))
    expected_scorer = str(config.get("scorer", ""))
    video_sets = dict(config.get("video_sets", {}))
    configured_stems = {Path(path).stem for path in video_sets}
    included_rows = 0
    skipped = []

    for folder in sorted((project_path / "labeled-data").glob("*")):
        if not folder.is_dir() or folder.name.startswith(".") or folder.name.endswith("_labeled"):
            continue
        label_path = folder / f"CollectedData_{expected_scorer}.h5"
        if not label_path.exists():
            skipped.append((folder.name, "matching scorer table not found"))
            continue
        try:
            labels = pd.read_hdf(label_path)
            bodypart_level = labels.columns.names.index("bodyparts")
            actual_bodyparts = list(dict.fromkeys(
                labels.columns.get_level_values(bodypart_level)
            ))
        except (KeyError, ValueError, OSError) as error:
            skipped.append((folder.name, f"unreadable label table: {error}"))
            continue
        # DLC's merger reindexes compatible columns into config order, so only
        # membership must match; rejecting a harmless HDF column-order change
        # formerly excluded every imported demo1 label table.
        if (set(actual_bodyparts) != set(expected_bodyparts) or
                len(actual_bodyparts) != len(expected_bodyparts)):
            skipped.append((folder.name, "bodyparts differ from config"))
            continue
        missing_images = [
            str(index[-1]) for index in labels.index
            if not (folder / str(index[-1])).exists()
        ]
        if missing_images:
            skipped.append((folder.name, f"{len(missing_images)} referenced images missing"))
            continue
        included_rows += len(labels)
        if folder.name not in configured_stems:
            # This path is an identifier for DLC's merger, not a file it opens.
            placeholder = project_path / "videos" / f"{folder.name}.mp4"
            video_sets[str(placeholder.absolute())] = {"crop": "0, 0, 0, 0"}
            configured_stems.add(folder.name)

    config["video_sets"] = video_sets
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return {
        "folders": len(configured_stems), "images": included_rows,
        "skipped": skipped,
    }


@contextmanager
def _single_animal_dlcrnet_paf(bodyparts: List[str], skeleton: List[List[str]]):
    """Supply DLC 3.0.1's missing single-animal DLCRNet PAF parameters."""
    from deeplabcut.pose_estimation_pytorch.config import PAFParameters
    from deeplabcut.pose_estimation_pytorch.config import pose as pose_config_module

    graph = _paf_graph_from_skeleton(bodyparts, skeleton)
    original = pose_config_module.build_pose_config_defaults

    def build_with_single_animal_paf(*args, **kwargs):
        """Inject Cheese3D PAF edges only for a DLCRNet call lacking them."""
        net_type = kwargs.get("net_type", args[0] if args else None)
        net_name = str(getattr(net_type, "value", net_type))
        if net_name.startswith("dlcrnet") and kwargs.get("paf_parameters") is None:
            kwargs["paf_parameters"] = PAFParameters(
                paf_graph=graph,
                num_limbs=len(graph),
                paf_edges_to_keep=list(range(len(graph))),
            )
        return original(*args, **kwargs)

    # DLC's PoseConfig.build resolves this module-level function at runtime.
    pose_config_module.build_pose_config_defaults = build_with_single_animal_paf
    try:
        yield graph
    finally:
        pose_config_module.build_pose_config_defaults = original


class _DLCProgressWriter:
    """Convert DLC3 tqdm frame output into atomic per-camera JSON progress."""

    def __init__(self, progress_file: str):
        self.progress_file = Path(progress_file)

    def write(self, text: str) -> int:
        matches = re.findall(r"(\d+)\s*/\s*(\d+)", text)
        if matches:
            completed, total = map(int, matches[-1])
            temporary = self.progress_file.with_suffix(".tmp")
            temporary.write_text(json.dumps({"completed": completed, "total": total}))
            temporary.replace(self.progress_file)
        return len(text)

    def flush(self) -> None:
        """Provide the file-like interface expected by tqdm."""


def _dlc_track_gpu_worker(config_path: str, videos: List[tuple[str, str]], output_dir: str,
                          gpu_id: str, batch_size: int,
                          snapshot_index: Optional[int], shuffle: int) -> int:
    """Analyze one camera subset in an isolated DLC3 process on one GPU."""
    isolate_worker_output()
    import deeplabcut as dlc
    print(f"DLC3 GPU {gpu_id} analyzing {len(videos)} video(s)", flush=True)
    for video, progress_file in videos:
        with redirect_stderr(_DLCProgressWriter(progress_file)):
            dlc.analyze_videos(
                config=config_path, videos=[video], destfolder=output_dir,
                save_as_csv=False, device=f"cuda:{gpu_id}", batch_size=batch_size,
                snapshot_index=snapshot_index, shuffle=shuffle,
            )
    return len(videos)


def _enforce_dlc3_project_config(config_path: Path,
                                 allowed_keys: set[str],
                                 schema_version: int = 0) -> List[str]:
    """Remove non-DLC3 root fields and select DLC3's PyTorch engine.

    Some imported projects accidentally mixed generated DLC2 pose settings into
    the project-level YAML. DLC3 rejects those extra fields; they are not valid
    project options and must not be forwarded into a new DLC3 training run.
    """
    config = yaml.safe_load(config_path.read_text()) or {}
    removed = [key for key in config if key not in allowed_keys]
    # Former behavior retained every unknown DLC2 training key at project root,
    # causing DLC3's strict Pydantic ProjectConfig validation to fail.
    config = {key: value for key, value in config.items() if key in allowed_keys}
    config["engine"] = "pytorch"
    # DLC package 3.x currently calls its internal YAML schema version 0; use
    # DLC's constant instead of assuming it matches the package major version.
    config["config_version"] = schema_version
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return removed


class DLCBackend(Pose2dBackend):
    def __init__(self,
                 name: str,
                 root_dir: Path,
                 videos: List[Path],
                 keypoints: List[KeypointConfig],
                 experimenter: str = "default",
                 date: Optional[str] = None,
                 crops: Optional[List[BoundingBox]] = None,
                 frames_per_video: int = 5,
                 skeleton: Optional[List[List[str]]] = None,
                 canonical_config_path: Optional[str | Path] = None,
                 source_project_path: Optional[str | Path] = None,
                 source_format: Optional[str] = None,
                 copy_source_images: bool = True):
        """Initialize DLC with a project-root canonical configuration file."""
        super().__init__()
        self.name = name
        self.root_dir = root_dir.absolute()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.experimenter = experimenter
        self.date = maybe(date, datetime.now().strftime("%Y-%m-%d"))
        self.videos = videos
        self.crops = maybe(crops, [[None, None, None, None] for _ in videos])
        self.keypoints = keypoints
        self.skeleton = list(skeleton or [])
        self.frames_per_video = frames_per_video
        # DLC still requires config.yaml inside its native project; this root
        # path is the user-editable source synchronized around native calls.
        self.canonical_config_path = (Path(canonical_config_path).absolute()
                                      if canonical_config_path else None)
        self.source_project_path = (Path(source_project_path).resolve()
                                    if source_project_path else None)
        self.source_format = source_format
        self._selected_snapshot_index: Optional[int] = None
        self._selected_shuffle: Optional[int] = None
        self._selected_snapshot_path: Optional[Path] = None

        # Check if project already exists, if not create it
        if not self.project_path.exists():
            if len(self.videos) == 0:
                # ensure_dlc3_config() below reads the project config that
                # only create() writes, so continuing without videos fails
                # later with an inscrutable FileNotFoundError.
                raise RuntimeError(
                    "Cannot create a new DLC project without videos: no "
                    "recordings matched the project's video_regex/sessions "
                    "configuration. Declare the session(s) under `sessions:` "
                    "in config.yaml and check the files under video_root."
                )
            self.create()
            if self.source_project_path is not None:
                # Seed labeled-data from a foreign (possibly non-DLC) raw
                # project before overwrite_config() folds compatible
                # labeled-data folders into video_sets.
                self._import_source_project(copy_images=copy_source_images)
            self._pull_canonical_config(self.config_path)
            self.overwrite_config()
            self.fix_video_symlinks()
        else:
            self._pull_canonical_config(self.config_path)
            self.overwrite_config()
            self.fix_video_symlinks()
        self.ensure_dlc3_config()
        self._push_canonical_config(self.config_path)

    @classmethod
    def from_existing(cls,
                      project_path: Path,
                      root_dir: Path,
                      videos: List[Path],
                      keypoints: List[KeypointConfig],
                      crops: Optional[List[BoundingBox]],
                      skeleton: Optional[List[List[str]]] = None):
        # copy project over
        def _ignore(src, names):
            if Path(src).name == "videos":
                return names
            else:
                return []
        if project_path.resolve() != (root_dir / project_path.name).resolve():
            shutil.copytree(project_path, root_dir / project_path.name,
                            ignore=_ignore,
                            dirs_exist_ok=True,
                            symlinks=True,
                            ignore_dangling_symlinks=True)
        # create links for video files
        for video in videos:
            abspath = video.resolve()
            relpath = os.path.relpath(abspath, root_dir / project_path.name / "videos")
            os.symlink(relpath, root_dir / project_path.name / "videos" / video.name)
        # create dlc project
        name, experimenter, date = dlc_folder_to_components(project_path)
        return cls(name=name,
                   root_dir=root_dir,
                   videos=videos,
                   keypoints=keypoints,
                   experimenter=experimenter,
                   date=date,
                   crops=crops,
                   skeleton=skeleton)

    @property
    def dlc_name(self):
        return f"{self.name}-{self.experimenter}-{self.date}"

    @property
    def project_path(self):
        return self.root_dir / self.dlc_name

    @property
    def config_path(self):
        return self.project_path / "config.yaml"

    def ensure_dlc3_config(self) -> None:
        """Make the active project a validator-clean DLC3 PyTorch project."""
        from deeplabcut.core.config import ProjectConfig
        from deeplabcut.core.config.versioning import CURRENT_CONFIG_VERSION
        removed = _enforce_dlc3_project_config(
            self.config_path, set(ProjectConfig.model_fields), CURRENT_CONFIG_VERSION
        )
        if removed:
            print(
                f"Removed {len(removed)} non-DLC3 project options from "
                f"{self.config_path}; using the DLC3 PyTorch engine."
            )

    def create(self):
        import deeplabcut as dlc
        if len(self.videos) == 0:
            raise RuntimeError(
                "Cannot create DLC project with no videos. "
                "Please add videos to your Cheese3D project folder and config file."
            )
        dlc.create_new_project(
            project=self.name,
            experimenter=self.experimenter,
            working_directory=str(self.root_dir.absolute()),
            videos=self.videos,
            copy_videos=False
        )

    def _import_source_project(self, copy_images: bool = True) -> dict:
        """Seed labeled-data from a DLC/Lightning Pose/SLEAP source project."""
        keypoint_names = [keypoint.label for keypoint in self.keypoints]
        records = read_foreign_records(
            self.source_format, self.source_project_path, keypoint_names
        )
        summary = write_dlc_records(
            records, self.project_path, self.experimenter, keypoint_names,
            copy_images=copy_images,
        )
        print(
            f"Imported {summary['records']} labeled frames from "
            f"{self.source_format} source into {self.project_path}"
        )
        return summary

    def overwrite_config(self):
        # load dlc config file
        dlc_config = OmegaConf.load(self.config_path)
        dlc_config.project_path = str(self.project_path.absolute())
        # overwrite videos
        videos = {}
        for (video, crop) in zip(self.videos, self.crops):
            if crop is None or any(c is None for c in crop):
                height, width = VideoFrames.get_dims(video)
                crop = (maybe(crop[0], 0),
                        maybe(crop[2], int(width)),
                        maybe(crop[1], 0),
                        maybe(crop[3], int(height)))
            else:
                crop = (crop[0], crop[2], crop[1], crop[3])
            video_path = self.project_path / "videos" / video.name
            videos[str(video_path.absolute())] = {
                "crop": ", ".join(map(str, crop))
            }
        dlc_config.video_sets = videos
        # overwrite bodyparts
        dlc_config.bodyparts = [kp.label for kp in self.keypoints]
        # Propagate Cheese3D's grouped skeleton instead of retaining DLC's
        # placeholder [[bodypart1, bodypart2], [objectA, bodypart3]] edges.
        dlc_config.skeleton = self.skeleton
        # overwrite number of frames to pick
        dlc_config.numframes2pick = self.frames_per_video
        # dump updated dlc config to disk
        OmegaConf.save(dlc_config, self.config_path, resolve=True)
        # The former assignment above intentionally remains the source of real
        # camera crops; now augment it with compatible imported label folders.
        _include_compatible_labeled_data(self.config_path)

    def fix_video_symlinks(self):
        videos = [Path(p) for p in reglob(r".*", str(self.project_path / "videos"))]
        project_videos = [p.name for p in self.videos]
        for video in videos:
            if video.name in project_videos:
                abspath = video.resolve()
                relpath = os.path.relpath(abspath, self.project_path / "videos")
                os.remove(video)
                os.symlink(relpath, video)

    def import_c3d_labels(self, videos: Dict[str, Path]):
        for name, path in videos.items():
            label_folder = self.project_path / "labeled-data" / name
            if not label_folder.exists():
                label_folder.mkdir(exist_ok=True, parents=True)
            images = reglob(r".*\.png", str(path))
            for image in images:
                src_image = Path(image)
                dst_image = label_folder / src_image.name
                if dst_image.exists():
                    os.remove(dst_image)
                relpath = os.path.relpath(src_image, label_folder)
                os.symlink(relpath, dst_image)
            annotations_yaml = path / "annotations.yaml"
            hdf = label_folder / f"CollectedData_{self.experimenter}.h5"
            csv = label_folder / f"CollectedData_{self.experimenter}.csv"
            if annotations_yaml.exists():
                with open(annotations_yaml, "r") as f:
                    annotations = yaml.safe_load(f)
                data_dict = {}
                for kp, files in annotations.items():
                    for file, coords in files.items():
                        # create index for this row
                        idx = ('labeled-data', str(label_folder.name), file)
                        # get x and y coordinates (could be null/None)
                        x_coord = coords[0][0]
                        y_coord = coords[0][1]
                        # create column keys for x and y
                        x_col = (self.experimenter, kp, 'x')
                        y_col = (self.experimenter, kp, 'y')
                        # store in data dictionary
                        if idx not in data_dict:
                            data_dict[idx] = {}
                        data_dict[idx][x_col] = x_coord
                        data_dict[idx][y_col] = y_coord
                # convert to dataframe
                index = pd.MultiIndex.from_tuples(list(data_dict.keys()))
                columns = pd.MultiIndex.from_tuples(
                    [(self.experimenter, bp, coord)
                     for bp in annotations.keys()
                     for coord in ['x', 'y']],
                    names=["scorer", "bodyparts", "coords"]
                )
                # create empty df with the right structure
                df = pd.DataFrame(index=index, columns=columns)
                # fill in the values
                for idx, values in data_dict.items():
                    for col, val in values.items():
                        df.loc[idx, col] = val
                # convert to float (this will convert None/null to NaN)
                df = df.astype(float)
                # write dataframe to disk
                if hdf.exists():
                    os.remove(hdf)
                df.to_hdf(hdf, key="df", mode="w")
                if csv.exists():
                    os.remove(csv)
                df.to_csv(csv)

    def export_c3d_labels(self, videos: Dict[str, Path]):
        for name, path in videos.items():
            label_folder = self.project_path / "labeled-data" / name
            if label_folder.exists():
                images = reglob(r".*\.png", str(label_folder))
                for image in images:
                    image = Path(image)
                    if image.resolve() != (path / image.name).resolve():
                        shutil.copy2(image, path)
                    relpath = os.path.relpath(path / image.name, label_folder)
                    os.remove(image)
                    os.symlink(relpath, image)
                hdf = label_folder / f"CollectedData_{self.experimenter}.h5"
                if hdf.exists():
                    df = pd.read_hdf(hdf)
                    annotations = {}
                    for kp in self.keypoints:
                        annotations[kp.label] = {}
                        coords_df = df[self.experimenter][kp.label] # type: ignore
                        for i in range(len(coords_df)):
                            pt = coords_df.iloc[i]
                            x, y = pt["x"], pt["y"]
                            x = None if pd.isna(x) else float(x)
                            y = None if pd.isna(y) else float(y)
                            file = pt.name[2]
                            annotations[kp.label][file] = [[x, y]]
                    with open(path / "annotations.yaml", "w") as f:
                        yaml.safe_dump(annotations, f)

    def extract_frames(self, videos: Optional[List[Path]] = None):
        import deeplabcut as dlc
        project_videos = maybe(videos, [p.name for p in self.videos])
        videos_list = [p for p in reglob(r".*", str(self.project_path / "videos"))
                         if Path(p).name in project_videos]
        dlc.extract_frames(config=self.config_path,
                           userfeedback=False,
                           crop=True,
                           videos_list=videos_list)

    def train(self, gpu, iterate_dataset: bool = True,
              training_settings: Optional[dict] = None):
        """Train the selected DLC3 PyTorch architecture with GUI overrides."""
        import deeplabcut as dlc
        # Re-read the root copy immediately before training so edits made while
        # Cheese3D is open are applied to DLC's required backend-local file.
        self._pull_canonical_config(self.config_path)
        self.overwrite_config()
        self.ensure_dlc3_config()
        label_summary = _include_compatible_labeled_data(self.config_path)
        print(
            f"DLC3 dataset selection: {label_summary['images']} labeled images "
            f"from {label_summary['folders']} folders"
        )
        if label_summary["skipped"]:
            print(
                "DLC3 skipped incompatible label folders: "
                + "; ".join(f"{name} ({reason})" for name, reason in label_summary["skipped"])
            )
        settings = training_settings or {}
        network_architecture = str(settings.get("network_architecture", "resnet_50"))
        available_models = tuple(dlc.pose_estimation_pytorch.available_models())
        if network_architecture not in available_models:
            raise ValueError(
                f"DLC3 model '{network_architecture}' is unavailable in this environment. "
                f"Available models: {', '.join(available_models)}"
            )
        train_fraction_percent = float(settings.get("train_fraction_percent", 95))
        if not 1 <= train_fraction_percent <= 99:
            raise ValueError("DLC training percentage must be between 1 and 99")
        train_fraction = train_fraction_percent / 100.0
        requested_shuffle = int(settings.get("training_shuffle", 1))
        if requested_shuffle <= 0:
            raise ValueError("DLC training shuffle must be positive")
        project_config = yaml.safe_load(self.config_path.read_text()) or {}
        previous_fractions = [float(value) for value in
                              project_config.get("TrainingFraction", [])]
        split_changed = not any(
            abs(value - train_fraction) < 1e-9 for value in previous_fractions
        )
        # DLC creates splits from the project-level TrainingFraction list. The
        # selected GUI percentage is persisted so later evaluation resolves the
        # same training/test split instead of silently returning to its old value.
        project_config["TrainingFraction"] = [train_fraction]
        self.config_path.write_text(yaml.safe_dump(project_config, sort_keys=False))
        self._push_canonical_config(self.config_path)
        training_datasets = reglob("iteration-[0-9]+", path=str(self.project_path / "training-datasets"))
        if iterate_dataset and len(training_datasets) > 0:
            dlc.merge_datasets(config=self.config_path)
        created_splits = None
        if split_changed:
            print(
                f"DLC3 training fraction changed from {previous_fractions or 'unset'} "
                f"to {[train_fraction]}; creating a new training-set shuffle."
            )
        if network_architecture.startswith("dlcrnet"):
            with _single_animal_dlcrnet_paf(
                [keypoint.label for keypoint in self.keypoints], self.skeleton
            ) as paf_graph:
                print(f"DLC3 DLCRNet PAF graph: {len(paf_graph)} Cheese3D skeleton edges")
                created_splits = dlc.create_training_dataset(
                    config=self.config_path, userfeedback=False,
                    net_type=network_architecture, augmenter_type="imgaug",
                    Shuffles=[requested_shuffle],
                )
        else:
            created_splits = dlc.create_training_dataset(
                config=self.config_path,
                userfeedback=False,
                # Formerly this was fixed to resnet_50; the Training tab now
                # selects any installed DLC3 model and may create shuffle 2+.
                net_type=network_architecture,
                augmenter_type="imgaug",
                Shuffles=[requested_shuffle],
            )
        created_shuffle = _shuffle_from_created_splits(
            created_splits, default=requested_shuffle
        )
        gpu_ids = [int(item.strip()) for item in str(gpu).split(",") if item.strip()]
        gpu_ids = _supported_training_gpus(network_architecture, gpu_ids)
        if gpu_ids:
            import torch
            # DLC3's DataParallel runner requires the generic ``cuda`` device,
            # then uses runner.gpus for placement. Set its primary CUDA device
            # first so a list such as [1, 0] does not leave base weights on 0.
            torch.cuda.set_device(gpu_ids[0])
        # DLC3 requires dot-separated override paths rather than a nested dict.
        augmentation = {
            "data.train.affine.rotation": settings.get("rotation", 30),
            "data.train.affine.scaling": [settings.get("scale_min", 0.5),
                                          settings.get("scale_max", 1.25)],
            # Every architecture in DLC3_PYTORCH_MODELS is bottom-up and so
            # samples crops from the full frame. Top-down networks, which take
            # data.train.top_down_crop instead, are excluded from the registry.
            "data.train.crop_sampling.width": settings.get("crop_width", 448),
            "data.train.crop_sampling.height": settings.get("crop_height", 448),
            "data.train.motion_blur": settings.get("motion_blur", True),
            "data.train.gaussian_noise": settings.get("gaussian_noise", 12.75),
            "runner.optimizer.params.lr": resolve_learning_rate(
                network_architecture, settings
            ),
            "runner.gpus": gpu_ids,
            "runner.eval_interval": settings.get("validate_every_n_epochs", 10),
        }
        print(f"DLC3 training model: {network_architecture}")
        print(f"DLC3 training shuffle: {created_shuffle}")
        print(
            f"DLC3 split: {train_fraction_percent:g}% train / "
            f"{100 - train_fraction_percent:g}% test"
        )
        print(f"DLC3 training settings: {settings}")
        dlc.train_network(config=self.config_path,
                          # DLC may create shuffle 2+ when a fraction changes;
                          # omitting this formerly made train_network look for 1.
                          shuffle=created_shuffle,
                          max_snapshots_to_keep=settings.get(
                              "max_snapshots_to_keep", 5
                          ),
                          # Passing None formerly resolved to cuda:0 while
                          # DataParallel scattered inputs to cuda:1. DLC's own
                          # runner documentation requires exactly "cuda" here.
                          device="cuda" if gpu_ids else "cpu",
                          epochs=settings.get("epochs"),
                          batch_size=settings.get("batch_size"),
                          save_epochs=settings.get("save_every_n_epochs"),
                          pytorch_cfg_updates=augmentation)
        dlc.evaluate_network(config=self.config_path,
                             # Evaluation formerly fell back to shuffle 1 even
                             # when create_training_dataset returned shuffle 3+,
                             # producing a misleading missing-metadata warning.
                             shuffles=[created_shuffle],
                             # DLC3 evaluation is single-device even after DDP
                             # training; use the first selected GPU explicitly.
                             device=f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
        self._publish_network_config()

    def _publish_network_config(self) -> None:
        """Publish the newest generated DLC3 network YAML in the project root."""
        if self.canonical_config_path is None:
            return
        candidates = list(self.project_path.rglob("pytorch_config.yaml"))
        if not candidates:
            return
        # DLC owns its generated in-model copy, while this stable root filename
        # makes the effective network settings easy to inspect and archive.
        newest = max(candidates, key=lambda path: path.stat().st_mtime)
        destination = self.canonical_config_path.with_name("dlc_network_config.yaml")
        shutil.copy2(newest, destination)

    def list_checkpoints(self) -> List[dict]:
        """List snapshots with their DLC split, shuffle, local index, and metrics."""
        import torch
        config = yaml.safe_load(self.config_path.read_text()) or {}
        iteration = config.get("iteration", 0)
        root = self.project_path / "dlc-models-pytorch" / f"iteration-{iteration}"
        records = []
        folder_pattern = re.compile(r"trainset(?P<fraction>\d+)shuffle(?P<shuffle>\d+)$")
        for train_folder in sorted(root.glob("*/train")) if root.exists() else []:
            match = folder_pattern.search(train_folder.parent.name)
            if match is None:
                continue
            # Match DLC's folder-local snapshot order: normal epochs first and
            # the best-validation snapshot last, regardless of its epoch.
            snapshots = sorted(
                train_folder.glob("snapshot*.pt"),
                key=lambda path: (
                    "-best-" in path.name,
                    int(re.search(r"(\d+)\.pt$", path.name).group(1)),
                ),
            )
            for snapshot_index, path in enumerate(snapshots):
                payload = torch.load(path, map_location="cpu", weights_only=False)
                metadata = payload.get("metadata", {})
                metrics = dict(metadata.get("metrics", {}))
                metrics.update({
                    name: value for name, value in metadata.get("losses", {}).items()
                    if name.startswith("eval.")
                })
                records.append({
                    "path": str(path), "epoch": metadata.get("epoch"),
                    "metrics": metrics, "shuffle": int(match.group("shuffle")),
                    "train_fraction_percent": int(match.group("fraction")),
                    "snapshot_index": snapshot_index,
                })
        records.sort(key=lambda item: (item["shuffle"], item["snapshot_index"]))
        return records

    def select_checkpoint(self, checkpoint: str | Path) -> None:
        """Map an explicit DLC3 snapshot path to DLC's ordered snapshot index."""
        records = self.list_checkpoints()
        paths = [Path(item["path"]).resolve() for item in records]
        selected = Path(checkpoint).resolve()
        if selected not in paths:
            raise FileNotFoundError(f"DLC3 checkpoint is unavailable: {checkpoint}")
        record = records[paths.index(selected)]
        self._selected_snapshot_index = int(record["snapshot_index"])
        self._selected_shuffle = int(record["shuffle"])
        self._selected_snapshot_path = selected

    def selected_result_identifiers(self) -> List[str]:
        """Identify Anipose outputs belonging to the GUI-selected DLC weights."""
        shuffle = getattr(self, "_selected_shuffle", None)
        snapshot = getattr(self, "_selected_snapshot_path", None)
        if shuffle is None or snapshot is None:
            return []
        return [f"shuffle{shuffle}", snapshot.stem]

    def track(self,
              videos: Dict[str, Path],
              output_dir: Path,
              calibration_path: Optional[Path] = None,
              tracking_settings: Optional[dict] = None) -> bool:
        """Track with the selected DLC3 checkpoint, device, and batch size."""
        import deeplabcut as dlc
        settings = tracking_settings or {}
        gpu_ids = [item.strip() for item in str(settings.get("gpu_ids", "0")).split(",")
                   if item.strip()]
        shuffle = int(settings.get("shuffle", getattr(self, "_selected_shuffle", None) or 1))
        device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu"
        print(f"DLC3 tracking settings: device={device}, "
              f"batch_size={settings.get('batch_size', 8)}, requested_gpus={gpu_ids}, "
              f"shuffle={shuffle}, snapshot_index={self._selected_snapshot_index}")
        output_dir.mkdir(parents=True, exist_ok=True)
        missing_videos = []
        for video in videos.values():
            video = Path(video)
            existing = list(output_dir.glob(f"{video.stem}*.h5"))
            if len(existing) == 0 or self._selected_snapshot_index is not None:
                missing_videos.append(video)
        if len(missing_videos) == 0:
            return True
        video_paths = [str(video.resolve()) for video in missing_videos]
        if len(gpu_ids) > 1 and len(video_paths) > 1:
            assignments = partition_videos_by_gpu(
                [Path(video) for video in video_paths], gpu_ids
            )
            print(f"DLC3 multi-GPU video assignments: {assignments}")
            context = multiprocessing.get_context("spawn")
            with tempfile.TemporaryDirectory(prefix="cheese3d-dlc-progress-") as progress_dir:
                progress_files = {
                    Path(video).stem: Path(progress_dir) / f"{Path(video).stem}.json"
                    for video in video_paths
                }
                pool = ProcessPoolExecutor(max_workers=len(assignments), mp_context=context)
                try:
                    futures = [pool.submit(
                        _dlc_track_gpu_worker, str(self.config_path),
                        [(str(video), str(progress_files[video.stem])) for video in assigned],
                        str(output_dir), gpu_id, int(settings.get("batch_size", 8)),
                        self._selected_snapshot_index, shuffle,
                    ) for gpu_id, assigned in assignments]
                    monitor_camera_progress(futures, progress_files)
                finally:
                    # A blocking context-manager shutdown formerly froze the GUI
                    # after all CUDA inference futures had already completed.
                    shutdown_completed_process_pool(pool)
            return True
        dlc.analyze_videos(config=str(self.config_path),
                           videos=video_paths,
                           destfolder=str(output_dir),
                           save_as_csv=False,
                           device=device,
                           batch_size=settings.get("batch_size", 8),
                           shuffle=shuffle,
                           # DLC3's PyTorch API uses the underscored parameter;
                           # the former DLC2 spelling is rejected downstream.
                           snapshot_index=self._selected_snapshot_index)

        return True

register_pose_backend("dlc", DLCBackend)
