import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

from cheese3d.registry import load_entry_points, registered_names

POSE_BACKEND_REGISTRY: Dict[str, Any] = {}
BUILTIN_POSE_BACKENDS = {
    "dlc": "cheese3d.backends.dlc",
    "eks": "cheese3d.backends.eks",
    "lightning_pose": "cheese3d.backends.lightning_pose",
}

def register_pose_backend(name: str, backend_cls):
    POSE_BACKEND_REGISTRY[name] = backend_cls

def check_pose_backend_conflicts():
    if (importlib.util.find_spec("deeplabcut") is not None and
        importlib.util.find_spec("lightning_pose") is not None):
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

    def train(self, gpu, iterate_dataset: bool = True):
        """Train the model using GPU ID `gpu`."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def track(self,
              videos: Dict[str, Path],
              output_dir: Path,
              calibration_path: Optional[Path] = None) -> bool:
        """Track videos into output_dir, returning True when handled by the backend."""
        return False
