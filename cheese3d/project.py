import os
import re
from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf
from rich import table, console
from typing import List, Dict

from cheese3d.config import MultiViewConfig, ProjectConfig, KeypointConfig
from cheese3d.utils import reglob

def find_videos(dir: Path,
                recording_regex: str,
                calibration_keys: Dict[str, str],
                recordings: List[Dict[str, str]],
                views: MultiViewConfig):
    videos = {}
    calibration_videos = {}
    for recording in recordings:
        if "name" in recording:
            session = recording["name"]
        else:
            raise RuntimeError("Recording entries must contain the 'name' key")
        grouped_videos = {}
        grouped_cal_videos = {}
        matches = [re.match(recording_regex, f)
                   for f in reglob(recording_regex, path=str(dir / session))]
        for view, cfg in views.items():
            for match in matches:
                if (match is None) or (match.group("view") != cfg.path):
                    continue
                if all(match.group(k) == v
                       for k, v in recording.items() if k != "name"):
                    group_key = (
                        session +
                        "/" +
                        match.group(0).replace(match.group("view"), "").split(os.sep)[-1]
                    )
                    if all(match.group(k) == v
                           for k, v in calibration_keys.items()):
                        group_dict = grouped_cal_videos
                    else:
                        group_dict = grouped_videos
                    if group_key in grouped_videos:
                        group_dict[group_key][view] = Path(match.group(0))
                    else:
                        group_dict[group_key] = {view: Path(match.group(0))}
        videos.update(**grouped_videos)
        calibration_videos.update(**grouped_cal_videos)

    return videos, calibration_videos

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
    calibrations: Dict[str, Dict[str, Path]]
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
    def from_cfg(cls, cfg: ProjectConfig, root: str | Path):
        root = Path(root)
        recordings, calibrations  = find_videos(
            dir=root / cfg.name / cfg.recording_root,
            recording_regex=cfg.video_regex,
            calibration_keys=cfg.calibration,
            recordings=cfg.recordings,
            views=cfg.views
        )

        return cls(name=cfg.name,
                   root=root,
                   recordings=recordings,
                   calibrations=calibrations,
                   keypoints=cfg.keypoints)

    @classmethod
    def from_path(cls, path: str | Path, cfg_dir = None, overrides = None):
        path = Path(path)
        cfg_file = path / "config.yaml"
        cfg = ProjectConfig.load(cfg_file, cfg_dir, overrides)

        return cls.from_cfg(cfg, path.parent) # type: ignore

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
        tab = table.Table("Label", "Group(s)", "View(s)", title="Project keypoints")
        for pt in self.keypoints:
            tab.add_row(pt.label, ", ".join(pt.groups), ", ".join(pt.views))
        pty.print(tab)
        # print recording infor
        tab = table.Table("Recording", "Files", title="Project recordings")
        for group, files in self.recordings.items():
            tab.add_row(group,
                        ",\n".join([f"{view}: {file.relative_to(self.path)}"
                                    for view, file in files.items()]))
        pty.print(tab)
