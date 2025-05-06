import re
from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf
from rich import table, console
from typing import List, Dict

from cheese3d.config import MultiViewConfig, ProjectConfig, KeypointConfig
from cheese3d.utils import reglob

def find_videos(dir: Path, recordings: List[str], views: MultiViewConfig):
    videos = {}
    for recording in recordings:
        grouped_videos = {}
        for view, cfg in views.items():
            matches = reglob(rf".*({cfg.path}).*\.avi", path=str(dir / recording))
            unique_matches = [re.sub(rf"(.*)({cfg.path})(.*)\.avi", r"\1_\3.avi", f)
                              for f in matches]
            for group, video in zip(unique_matches, matches):
                group = group.split("/")[-1]
                if group in grouped_videos:
                    grouped_videos[group][view] = video
                else:
                    grouped_videos[group] = {view: video}
        videos.update(**grouped_videos)

    return videos

@dataclass
class Ch3DProject:
    """
    A Cheese3D project.

    Arguments:
        - `name`: the name of the project
        - `root`: root directory under which the project folder should be made
        - `recordings`: a list of recordings where each entry is video files
            organized by camera view
        - `keypoints`: a list of `KeypointConfig`s to track in this project
    """
    name: str
    root: Path
    recordings: Dict[str, Dict[str, Path]]
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
        recordings = find_videos(Path(cfg.root) / cfg.name / cfg.recording_root,
                                 cfg.recordings,
                                 cfg.videos)
        return cls(cfg.name, Path(cfg.root), recordings, keypoints=cfg.keypoints)

    @classmethod
    def from_path(cls, path: str | Path, cfg_dir = None, overrides = None):
        cfg_file = Path(path) / "config.yaml"
        cfg = ProjectConfig.load(cfg_file, cfg_dir, overrides)

        return cls.from_cfg(cfg) # type: ignore

    def summarize(self):
        pty = console.Console()
        # print basic info
        tab = table.Table(title="Cheese3D project info")
        tab.add_column("Key")
        tab.add_column("Value")
        tab.add_row("Name", self.name)
        tab.add_row("Root Path", str(self.root))
        pty.print(tab)
        # print keypoint info
        tab = table.Table("Label", "Groups", "Views", title="Project keypoints")
        for pt in self.keypoints:
            tab.add_row(pt.label, ", ".join(pt.groups), ", ".join(pt.views))
        pty.print(tab)
        # print recording infor
        tab = table.Table("Recording", "Files", title="Project recordings")
        for name, files in self.recordings.items():
            tab.add_row(name, ",\n".join([f"{view}: {Path(file).relative_to(self.path)}"
                                         for view, file in files.items()]))
        pty.print(tab)
