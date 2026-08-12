import importlib
import importlib.util
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cheese3d.registry import load_entry_points, registered_names
from tqdm.auto import tqdm


def isolate_worker_output() -> None:
    """Detach this multiprocessing worker's stdout/stderr from its inherited pipe.

    When Cheese3D runs under Textual Serve, the process's real fd 1 (stdout)
    is the pipe textual-serve's web driver uses to send rendered frames to
    the browser. multiprocessing.get_context("spawn") workers still inherit
    that fd directly (close_fds only affects fds >= 3), so any stdout/stderr
    write from a worker -- our own prints, a nested subprocess.run() with no
    output capture, or FFmpeg/OpenCV/PyTorch's own C-level writes -- lands in
    that same shared pipe. With several worker processes writing concurrently
    (one per camera/GPU), this corrupts Textual's framing protocol on that
    pipe, silently breaking the browser's connection with no way to recover
    short of restarting the whole GUI -- confirmed as the actual cause of a
    real reported freeze during video generation and (structurally identical)
    tracking, which was otherwise indistinguishable from a Python deadlock.
    Called first thing inside each ProcessPoolExecutor/multiprocessing worker
    function; subsequent subprocess.run() calls with no explicit stdout/
    stderr then inherit this safe redirection instead of the original pipe.
    """
    devnull = open(os.devnull, "w")
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)
    sys.stdout = devnull
    sys.stderr = devnull


# Every reader below returns/consumes the same normalized shape, so any
# backend's "seed my labels from a foreign project" path can share one
# implementation per source format instead of each backend re-parsing every
# other framework's on-disk layout itself:
#   dict[(folder_name, image_name)] -> (image_path: Path, points: List[[x, y]])
# where `points` is ordered to match the caller's keypoint_names, and missing
# values are represented as float('nan').
Records = Dict[Tuple[str, str], Tuple[Path, List[List[float]]]]


def read_dlc_records(project_path: str | Path, keypoint_names: Sequence[str]) -> Records:
    """Read compatible DLC label folders without importing DeepLabCut itself."""
    records: Records = {}
    labeled_root = Path(project_path) / "labeled-data"
    for folder in sorted(labeled_root.glob("*")) if labeled_root.is_dir() else []:
        tables = sorted(folder.glob("CollectedData_*.h5"))
        if not tables:
            continue
        try:
            labels = pd.read_hdf(tables[0])
        except (KeyError, OSError, ValueError):
            continue
        # A keypoint absent from this folder's own schema becomes NaN below
        # (per-keypoint) rather than discarding the whole folder here; callers
        # needing every point present for every frame can filter afterward.
        scorer = labels.columns.get_level_values(0)[0]
        for index, row in labels.iterrows():
            image_name = str(index[-1] if isinstance(index, tuple) else index)
            image_path = folder / Path(image_name).name
            if not image_path.is_file():
                continue
            points = []
            for keypoint in keypoint_names:
                try:
                    points.append([
                        float(row[(scorer, keypoint, "x")]),
                        float(row[(scorer, keypoint, "y")]),
                    ])
                except KeyError:
                    points.append([float("nan"), float("nan")])
            records[(folder.name, image_path.name)] = (image_path, points)
    return records


def read_lp_records(project_path: str | Path, keypoint_names: Sequence[str]) -> Records:
    """Read a Lightning Pose project's CollectedData.csv into normalized records.

    Unlike DLC, LP always labels pre-extracted images (never raw video frame
    indices), so paths in the CSV already resolve directly under ``data/``.
    """
    project_path = Path(project_path)
    csv_path = project_path / "data" / "CollectedData.csv"
    if not csv_path.is_file():
        return {}
    labels = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    scorer = labels.columns.get_level_values(0)[0]
    records: Records = {}
    for index_value, row in labels.iterrows():
        relative_path = Path(str(index_value))
        image_path = project_path / "data" / relative_path
        if not image_path.is_file():
            continue
        points = []
        for keypoint in keypoint_names:
            try:
                points.append([
                    float(row[(scorer, keypoint, "x")]),
                    float(row[(scorer, keypoint, "y")]),
                ])
            except KeyError:
                points.append([float("nan"), float("nan")])
        records[(relative_path.parent.name, relative_path.name)] = (image_path, points)
    return records


def read_sleap_records(project_path: str | Path, keypoint_names: Sequence[str]) -> Records:
    """Read a raw SLEAP project's labels.slp into normalized records.

    SLEAP labels can be backed by either an image sequence (Cheese3D's own
    SLP packages) or a real video file (common for a project labeled directly
    in the SLEAP GUI). Image-sequence frames are referenced directly; video
    frames are decoded and written out as real image files once, since every
    downstream writer (DLC/LP) needs an actual image file to copy, not a
    frame index into a video it can't necessarily read compatibly.
    """
    import sleap_io as sio
    from PIL import Image

    project_path = Path(project_path)
    labels_path = project_path / "labels.slp"
    if not labels_path.is_file():
        candidates = sorted(project_path.glob("*.slp"))
        if not candidates:
            return {}
        labels_path = candidates[0]

    labels = sio.load_slp(str(labels_path), open_videos=True)
    node_names = [node.name for node in labels.skeletons[0].nodes] if labels.skeletons else []
    extracted_root = project_path / ".cheese3d_extracted_frames"

    records: Records = {}
    for labeled_frame in labels:
        if not labeled_frame.instances:
            continue
        video = labeled_frame.video
        filenames = video.filename
        if isinstance(filenames, list):
            # Image-sequence video: reference the already-existing frame directly.
            image_path = Path(filenames[labeled_frame.frame_idx])
            folder_name = image_path.parent.name
            image_name = image_path.name
        else:
            # Real video file: decode this one frame and persist it once so
            # every downstream writer has an ordinary image file to copy.
            video_stem = Path(filenames).stem
            folder_name = video_stem
            image_name = f"frame_{labeled_frame.frame_idx:06d}.png"
            image_path = extracted_root / video_stem / image_name
            if not image_path.is_file():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                frame = np.asarray(video[labeled_frame.frame_idx])
                if frame.ndim == 3 and frame.shape[-1] == 1:
                    frame = frame[..., 0]
                Image.fromarray(frame).save(image_path)

        instance = labeled_frame.instances[0]
        instance_points = dict(zip(node_names, instance.numpy().tolist()))
        points = []
        for keypoint in keypoint_names:
            xy = instance_points.get(keypoint)
            if xy is None:
                points.append([float("nan"), float("nan")])
            else:
                points.append([float(xy[0]), float(xy[1])])
        records[(folder_name, image_name)] = (image_path, points)
    return records


SOURCE_FORMAT_READERS = {
    "dlc": read_dlc_records,
    "lightning_pose": read_lp_records,
    "sleap": read_sleap_records,
}


def read_foreign_records(source_format: str, source_project_path: str | Path,
                         keypoint_names: Sequence[str]) -> Records:
    """Dispatch to the matching raw-project reader for a source framework.

    This is the shared entry point every backend's "import a foreign project"
    constructor option should call: the source format (which raw layout to
    parse) is independent of the active backend (what gets written), so any
    environment can import a DLC, Lightning Pose, or SLEAP source project.
    """
    reader = SOURCE_FORMAT_READERS.get(source_format)
    if reader is None:
        raise ValueError(
            f"Unsupported source framework '{source_format}'; expected one of "
            f"{sorted(SOURCE_FORMAT_READERS)}"
        )
    return reader(source_project_path, keypoint_names)


def write_dlc_records(records: Records, project_path: str | Path, experimenter: str,
                      keypoint_names: Sequence[str], copy_images: bool = True) -> dict:
    """Write normalized records into DLC's native labeled-data/CollectedData layout.

    This produces exactly the format DLC's own dataset merger
    (``_include_compatible_labeled_data``) and training pipeline already read,
    so a project seeded this way is trainable without any further conversion.
    """
    project_path = Path(project_path)
    labeled_root = project_path / "labeled-data"
    by_folder: Dict[str, List[Tuple[str, Path, List[List[float]]]]] = {}
    for (folder_name, image_name), (image_path, points) in records.items():
        by_folder.setdefault(folder_name, []).append((image_name, Path(image_path), points))

    columns = pd.MultiIndex.from_tuples([
        (experimenter, keypoint, coord)
        for keypoint in keypoint_names for coord in ("x", "y")
    ], names=["scorer", "bodyparts", "coords"])

    written_folders = 0
    written_images = 0
    for folder_name, items in sorted(by_folder.items()):
        label_dir = labeled_root / folder_name
        label_dir.mkdir(parents=True, exist_ok=True)
        index_tuples = []
        rows = []
        for image_name, image_path, points in sorted(items, key=lambda item: item[0]):
            destination = label_dir / Path(image_name).name
            if copy_images and image_path.resolve() != destination.resolve():
                shutil.copy2(image_path, destination)
                written_images += 1
            index_tuples.append(("labeled-data", folder_name, destination.name))
            rows.append([value for point in points for value in point])
        table = pd.DataFrame(
            rows, index=pd.MultiIndex.from_tuples(index_tuples), columns=columns
        )
        hdf = label_dir / f"CollectedData_{experimenter}.h5"
        csv = label_dir / f"CollectedData_{experimenter}.csv"
        table.to_hdf(hdf, key="df", mode="w")
        table.to_csv(csv)
        written_folders += 1
    return {"folders": written_folders, "images": written_images, "records": len(records)}


def write_lp_records(records: Records, lp_project_path: str | Path,
                     keypoint_names: Sequence[str], copy_images: bool = True,
                     scorer: str = "lightning_pose") -> dict:
    """Write normalized records into Lightning Pose's native CollectedData.csv.

    Images are copied beneath ``<lp_project_path>/data/labeled-data`` so the
    project remains usable if the source project is later moved, matching
    ``convert_dlc_labels_to_lightning_pose``'s existing DLC-source behavior.
    """
    lp_project_path = Path(lp_project_path)
    data_dir = lp_project_path / "data"
    output_labeled_data = data_dir / "labeled-data"
    output_labeled_data.mkdir(parents=True, exist_ok=True)
    # A video directory is required by Lightning Pose's data-path validation
    # even for a fully supervised model with no unlabeled video losses.
    (data_dir / "videos").mkdir(parents=True, exist_ok=True)

    columns = pd.MultiIndex.from_tuples([
        (scorer, keypoint, coord)
        for keypoint in keypoint_names for coord in ("x", "y")
    ], names=["scorer", "bodyparts", "coords"])

    index_values = []
    rows = []
    copied_images = 0
    for (folder_name, image_name), (image_path, points) in sorted(records.items()):
        image_path = Path(image_path)
        destination = output_labeled_data / folder_name / Path(image_name).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if copy_images and image_path.resolve() != destination.resolve():
            shutil.copy2(image_path, destination)
            copied_images += 1
        index_values.append(str(Path("labeled-data") / folder_name / destination.name))
        rows.append([value for point in points for value in point])

    table = pd.DataFrame(rows, index=pd.Index(index_values, name=None), columns=columns)
    output_csv = data_dir / "CollectedData.csv"
    table.to_csv(output_csv)
    return {
        "data_dir": data_dir,
        "csv_file": output_csv,
        "num_frames": len(table),
        "num_keypoints": len(keypoint_names),
        "keypoint_names": list(keypoint_names),
        "copied_images": copied_images,
    }


POSE_BACKEND_REGISTRY: Dict[str, Any] = {}
BUILTIN_POSE_BACKENDS = {
    "dlc": "cheese3d.backends.dlc",
    "eks": "cheese3d.backends.eks",
    "lightning_pose": "cheese3d.backends.lightning_pose",
    "sleap": "cheese3d.backends.sleap",
}


def partition_videos_by_gpu(videos: List[Path], gpu_ids: List[str]) -> List[tuple[str, List[Path]]]:
    """Assign videos by estimated frame count to balance inference GPU work."""
    if not gpu_ids:
        return []
    videos = [Path(video) for video in videos]

    def _work_units(video: Path) -> int:
        """Read a cheap container frame-count estimate without decoding frames."""
        if not video.is_file():
            return 1
        try:
            import cv2
            capture = cv2.VideoCapture(str(video))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if frames > 0:
                return frames
        except Exception:
            # Corrupt/unusual containers still receive a deterministic file-size
            # estimate so one large camera does not monopolize a GPU by accident.
            pass
        return max(1, video.stat().st_size)

    groups = [{"gpu": gpu, "videos": [], "work": 0} for gpu in gpu_ids]
    original_order = {video: index for index, video in enumerate(videos)}
    work_units = {video: _work_units(video) for video in videos}
    # Longest-processing-time scheduling reduces the idle tail while equal-size
    # or synthetic test videos retain the historical round-robin assignment.
    weighted = sorted(enumerate(videos), key=lambda item: (-work_units[item[1]], item[0]))
    for _, video in weighted:
        group = min(enumerate(groups), key=lambda item: (item[1]["work"], item[0]))[1]
        work = work_units[video]
        group["videos"].append(video)
        group["work"] += work
    return [
        (str(group["gpu"]), sorted(group["videos"], key=original_order.get))
        for group in groups if group["videos"]
    ]


def monitor_camera_progress(futures, progress_files: Dict[str, Path],
                            unit: str = "batch") -> None:
    """Render one live progress bar per camera from atomic JSON status files."""
    bars = {
        name: tqdm(total=1, desc=name, unit=unit, position=index,
                   dynamic_ncols=True, leave=True)
        for index, name in enumerate(progress_files)
    }
    try:
        while not all(future.done() for future in futures):
            for name, path in progress_files.items():
                if not path.is_file():
                    continue
                try:
                    status = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                bar = bars[name]
                bar.total = max(1, int(status.get("total", 1)))
                completed = min(bar.total, int(status.get("completed", 0)))
                if completed > bar.n:
                    bar.update(completed - bar.n)
                bar.refresh()
            time.sleep(0.25)
        for future in futures:
            future.result()
        for name, bar in bars.items():
            path = progress_files[name]
            if path.is_file():
                status = json.loads(path.read_text())
                bar.total = max(1, int(status.get("total", 1)))
            if bar.n < bar.total:
                bar.update(bar.total - bar.n)
    finally:
        for bar in bars.values():
            bar.close()


def shutdown_completed_process_pool(pool) -> None:
    """Retire completed CUDA workers without blocking on interpreter teardown.

    CUDA children can complete their futures but remain alive during the
    executor's normal ``shutdown(wait=True)``. Results are collected before this
    helper is called; terminating only idle survivors lets the GUI return.
    """
    processes = list(getattr(pool, "_processes", {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2)

def register_pose_backend(name: str, backend_cls):
    POSE_BACKEND_REGISTRY[name] = backend_cls

def _is_installed(distribution: str) -> bool:
    """Return whether package-manager metadata exists for a distribution."""
    try:
        importlib.metadata.version(distribution)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def check_pose_backend_conflicts():
    """Reject environments that intentionally install both backend packages.

    Distribution metadata is used instead of module discovery because Pixi may
    leave an empty namespace or orphan source directory while changing features;
    those remnants should not make a correctly isolated environment unusable.
    """
    # Former detection used find_spec(), which treats stale Pixi directories as
    # installed packages even after their distribution metadata is removed.
    # if (importlib.util.find_spec("deeplabcut") is not None and
    #     importlib.util.find_spec("lightning_pose") is not None):
    installed = {
        "DLC": _is_installed("deeplabcut"),
        "Lightning Pose": _is_installed("lightning-pose"),
        "SLEAP": _is_installed("sleap"),
    }
    active = [name for name, present in installed.items() if present]
    if len(active) > 1:
        raise RuntimeError(
            f"Pose backends are mutually exclusive but found {', '.join(active)}. "
            "Use the dedicated Pixi environment for exactly one backend."
        )


def active_pose_backend() -> Optional[str]:
    """Return the backend_type whose package is actually installed here.

    Pose backend environments are mutually exclusive (see
    check_pose_backend_conflicts), so whichever distribution is importable
    reliably identifies which Pixi environment Cheese3D was launched in.
    Returns None when none are installed (e.g. the bare project-management
    environment), in which case callers should keep their own default.
    """
    for distribution, backend_type in (
        ("deeplabcut", "dlc"),
        ("lightning-pose", "lightning_pose"),
        ("sleap", "sleap"),
    ):
        if _is_installed(distribution):
            return backend_type
    return None

def load_builtin_pose_backend(name: str):
    module = BUILTIN_POSE_BACKENDS.get(name)
    if module is not None:
        importlib.import_module(module)

def get_pose_backend_class(name: str):
    check_pose_backend_conflicts()
    if name not in POSE_BACKEND_REGISTRY:
        load_builtin_pose_backend(name)
    if name not in POSE_BACKEND_REGISTRY:
        load_entry_points("cheese3d.pose_backends", POSE_BACKEND_REGISTRY)
    if name not in POSE_BACKEND_REGISTRY:
        raise RuntimeError(f"Unrecognized model backend {name}. "
                           f"Supported backends are: {registered_names(POSE_BACKEND_REGISTRY)}")

    return POSE_BACKEND_REGISTRY[name]

class Pose2dBackend:
    name: str

    def _pull_canonical_config(self, backend_path: str | Path) -> None:
        """Copy the project-root canonical config into its backend-native path."""
        canonical = getattr(self, "canonical_config_path", None)
        backend_path = Path(backend_path)
        if canonical is not None and Path(canonical).is_file():
            backend_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(canonical, backend_path)

    def _push_canonical_config(self, backend_path: str | Path) -> None:
        """Publish a validated backend config beside Cheese3D's main config."""
        canonical = getattr(self, "canonical_config_path", None)
        backend_path = Path(backend_path)
        if canonical is not None and backend_path.is_file():
            canonical = Path(canonical)
            canonical.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backend_path, canonical)

    @property
    def project_path(self):
        """Return the path to the actual project folder."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @classmethod
    def from_existing(cls, root_dir: Path, project_path: Path, *args, **kwargs):
        """Import existing backend project into Cheese3D project."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def import_c3d_labels(self, videos: Dict[str, Path]):
        """Import Cheese3D label file and store backend file."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def export_c3d_labels(self, videos: Dict[str, Path]):
        """Export backend labels on disk to Cheese3D labels."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def extract_frames(self, videos: Optional[List[Path]] = None):
        """Extract frames from videos."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def train(self, gpu, iterate_dataset: bool = True,
              training_settings: Optional[dict] = None):
        """Train the model using GPU ID `gpu`."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def list_checkpoints(self) -> List[dict]:
        """Return selectable checkpoints and their available validation metrics."""
        return []

    def select_checkpoint(self, checkpoint: str | Path) -> None:
        """Select an explicit checkpoint for subsequent pose tracking."""
        raise NotImplementedError("This backend does not expose checkpoint selection.")

    def selected_result_identifiers(self) -> List[str]:
        """Return filename terms identifying outputs from the selected checkpoint."""
        return []

    def track(self,
              videos: Dict[str, Path],
              output_dir: Path,
              calibration_path: Optional[Path] = None,
              tracking_settings: Optional[dict] = None) -> bool:
        """Track videos into output_dir, returning True when handled by the backend."""
        return False
