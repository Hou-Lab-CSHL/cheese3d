import importlib
import importlib.util
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cheese3d.registry import load_entry_points, registered_names
from tqdm.auto import tqdm

POSE_BACKEND_REGISTRY: Dict[str, Any] = {}
BUILTIN_POSE_BACKENDS = {
    "dlc": "cheese3d.backends.dlc",
    "eks": "cheese3d.backends.eks",
    "lightning_pose": "cheese3d.backends.lightning_pose",
}


def partition_videos_by_gpu(videos: List[Path], gpu_ids: List[str]) -> List[tuple[str, List[Path]]]:
    """Assign independent camera videos round-robin across selected GPUs."""
    if not gpu_ids:
        return []
    groups = [(gpu, []) for gpu in gpu_ids]
    for index, video in enumerate(videos):
        groups[index % len(groups)][1].append(Path(video))
    return [(gpu, assigned) for gpu, assigned in groups if assigned]


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

def register_pose_backend(name: str, backend_cls):
    POSE_BACKEND_REGISTRY[name] = backend_cls

def check_pose_backend_conflicts():
    """Reject environments that intentionally install both backend packages.

    Distribution metadata is used instead of module discovery because Pixi may
    leave an empty namespace or orphan source directory while changing features;
    those remnants should not make a correctly isolated environment unusable.
    """
    def _is_installed(distribution: str) -> bool:
        """Return whether package-manager metadata exists for a distribution."""
        try:
            importlib.metadata.version(distribution)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

    # Former detection used find_spec(), which treats stale Pixi directories as
    # installed packages even after their distribution metadata is removed.
    # if (importlib.util.find_spec("deeplabcut") is not None and
    #     importlib.util.find_spec("lightning_pose") is not None):
    if _is_installed("deeplabcut") and _is_installed("lightning-pose"):
        raise RuntimeError("DLC and Lightning Pose are mutually exclusive Cheese3D "
                           "backend extras. Install only one of `cheese3d[dlc]` or "
                           "`cheese3d[lightning-pose]` in an environment.")

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

    def track(self,
              videos: Dict[str, Path],
              output_dir: Path,
              calibration_path: Optional[Path] = None,
              tracking_settings: Optional[dict] = None) -> bool:
        """Track videos into output_dir, returning True when handled by the backend."""
        return False
