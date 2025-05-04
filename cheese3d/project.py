from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf
from rich import table, console
from typing import List

from cheese3d.config import ProjectConfig, KeypointConfig

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
    keypoints: List[KeypointConfig]

    @property
    def path(self):
        return self.root / self.name

    @staticmethod
    def initialize(name: str, root: str | Path):
        location = Path(root) / name
        if location.exists():
            raise RuntimeError(f"Project {name} already exists under {root}")
        # create project directory
        location.mkdir(parents=True)
        # create a empty configuration file
        cfg = OmegaConf.structured(ProjectConfig)
        cfg.name = name
        with location / "config.yaml" as f:
            OmegaConf.save(cfg, f)

    @classmethod
    def from_cfg(cls, cfg: ProjectConfig):
        return cls(cfg.name, Path(cfg.root), keypoints=cfg.keypoints)

    @classmethod
    def from_path(cls, path: str | Path, cfg_dir = None, overrides = None):
        cfg_file = Path(path) / "config.yaml"
        cfg = ProjectConfig.load(cfg_file, cfg_dir, overrides)

        return cls.from_cfg(cfg) # type: ignore

    def summarize(self):
        tab = table.Table(title="Cheese3D project info")
        tab.add_column("Key")
        tab.add_column("Value")
        tab.add_row("Name", self.name)
        tab.add_row("Root Path", str(self.root))
        tab.add_row("Keypoints", ", ".join([pt.label for pt in self.keypoints]))

        console.Console().print(tab)
