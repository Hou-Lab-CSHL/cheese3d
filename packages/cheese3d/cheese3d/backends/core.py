from pathlib import Path
from typing import Any, Dict, List, Optional

from cheese3d.registry import load_entry_points, registered_names

POSE_BACKEND_REGISTRY: Dict[str, Any] = {}

def register_pose_backend(name: str, backend_cls):
    POSE_BACKEND_REGISTRY[name] = backend_cls

def get_pose_backend_class(name: str):
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
