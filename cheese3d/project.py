from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf

from cheese3d.config import ProjectConfig

@dataclass
class Ch3DProject:
    """
    A Cheese3D project.

    Arguments:
        - `name`: the name of the project
        - `root`: root directory under which the project folder should be made
    """
    name: str
    root: Path

    @property
    def path(self):
        return self.root / self.name

    def initialize(self):
        if self.path.exists():
            raise RuntimeError(f"Project {self.name} already exists under {self.root}")
        # create project directory
        self.path.mkdir(parents=True)
        # create a empty configuration file
        cfg = OmegaConf.structured(ProjectConfig)
        with self.path / "config.yaml" as f:
            OmegaConf.save(cfg, f)
