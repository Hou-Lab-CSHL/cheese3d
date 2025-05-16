from pathlib import Path
from typing import Dict

class Pose2dBackend:
    name: str

    @classmethod
    def from_existing(cls, root_dir: Path, project_path: Path, *args, **kwargs):
        """Import existing backend project into Cheese3D project."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def import_c3d_labels(self):
        """Import Cheese3D label file and store backend file."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def export_c3d_labels(self, videos: Dict[str, Path]):
        """Export backend labels on disk to Cheese3D labels."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def extract_frames(self):
        """Extract frames from videos."""
        raise NotImplementedError("This method should be implemented by subclasses.")
