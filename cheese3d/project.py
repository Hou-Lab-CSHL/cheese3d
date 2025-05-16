import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from rich import table, console
from rich import print as rprint
from typing import List, Dict, Optional, Any
from collections import namedtuple

from cheese3d.config import MultiViewConfig, ProjectConfig, KeypointConfig
from cheese3d.synchronize.core import SyncConfig, SyncPipeline
from cheese3d.synchronize.readers import VideoSyncReader, get_ephys_reader
from cheese3d.utils import reglob

class RecordingKey(namedtuple("RecordingKey", ["session", "name", "attributes"])):
    __slots__ = () # prevent __dict__ creation since subclassing namedtuple

    def __new__(cls, session: str, name: str, **attributes):
        return super().__new__(cls, session, name, frozenset(attributes.items()))

    def __eq__(self, other):
        return (self.name == other.name) and self.matches(other)

    def __hash__(self):
        # we assume that self.name contains all the info in self.attributes
        # so it is never the case that two RecordingKeys with the same name
        # but different attributes are equal (in practice, not theory)
        return hash((self.session, self.name))

    def as_str(self):
        return ("(session: " + self.session + ", " +
                "name: " + self.name + ", " +
                ", ".join([f"{k}: {v}" for k, v in self.attributes]) + ")")

    def matches(self, other):
        is_matched = (self.session == other.session)
        other_attributes = dict(other.attributes)
        for k, v in self.attributes:
            if k in other_attributes:
                is_matched &= (v == other_attributes[k])

        return is_matched

def group_by_session(recordings: Dict[RecordingKey, Any]):
    sessions = set(r.session for r in recordings.keys())

    return {session: {k: v for k, v in recordings.items() if k.session == session}
            for session in sessions}

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
        for view, video_cfg in views.items():
            for match in matches:
                if (match is None) or (match.group("view") != video_cfg.path):
                    continue
                if all(match.group(k) == v
                       for k, v in recording.items() if k != "name"):
                    group_name = Path(match.group(0)
                                           .replace(match.group("view"), "")).stem
                    group_key = RecordingKey(session,
                                             group_name,
                                             **{k: v for k, v in match.groupdict().items()
                                                     if k != "view"})
                    if all(match.group(k) == v
                           for k, v in calibration_keys.items()):
                        group_dict = grouped_cal_videos
                    else:
                        group_dict = grouped_videos
                    if group_key in group_dict:
                        group_dict[group_key][view] = Path(match.group(0))
                    else:
                        group_dict[group_key] = {view: Path(match.group(0))}
        videos.update(grouped_videos)
        calibration_videos.update(grouped_cal_videos)

    return videos, calibration_videos

def find_ephys(dir: Path, ephys_regex: str, recordings: Dict[RecordingKey, Dict[str, Path]]):
    ephys = {}
    grouped_recordings = group_by_session(recordings)

    for session, session_recordings in grouped_recordings.items():
        matches = [re.match(ephys_regex, f)
                   for f in reglob(ephys_regex, path=str(dir / session))]
        ephys_keys = [RecordingKey(session, m.group(0),
                                   **{k: v for k, v in m.groupdict().items()})
                      for m in matches if m is not None]
        # warn if there are duplicate keys
        if len(ephys_keys) != len(set(ephys_keys)):
            rprint("[bold red]WARNING:[/bold red] "
                   f"Duplicate matches found for ephys recordings in {session=}."
                   "Ephys recordings will by matched to videos in alphabetical order.")
        for recording in session_recordings.keys():
            for key in ephys_keys:
                if recording.matches(key):
                    # pop key out of ephys_keys
                    ephys_key = ephys_keys.pop(ephys_keys.index(key))
                    merged_key = RecordingKey(session, recording.name, **dict(ephys_key.attributes))
                    ephys[merged_key] = dir / session / ephys_key.name
                    break

    return ephys

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
    fps: int
    recordings: Dict[RecordingKey, Dict[str, Path]]
    calibrations: Dict[RecordingKey, Dict[str, Path]]
    view_config: MultiViewConfig
    keypoints: List[KeypointConfig]
    ephys_recordings: Optional[Dict[RecordingKey, Path]] = None
    ephys_param: Optional[Dict[str, Any]] = None
    sync: SyncConfig = field(
        default_factory=lambda: SyncConfig(["crosscorr", "regression", "sample_rate"])
    )

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
        cfg = ProjectConfig.default()
        cfg.name = name
        with location / "config.yaml" as f:
            OmegaConf.save(cfg, f)

    @classmethod
    def from_cfg(cls, cfg: ProjectConfig, root: str | Path):
        root = Path(root)
        recordings, calibrations = find_videos(
            dir=root / cfg.name / cfg.recording_root,
            recording_regex=ProjectConfig.build_regex(cfg.video_regex),
            calibration_keys=cfg.calibration,
            recordings=cfg.recordings,
            views=cfg.views
        )
        if cfg.ephys_regex and cfg.ephys_root and cfg.ephys_param:
            ephys = find_ephys(
                dir=root / cfg.name / cfg.ephys_root,
                ephys_regex=ProjectConfig.build_regex(cfg.ephys_regex),
                recordings=recordings
            )
        elif cfg.ephys_regex or cfg.ephys_root or cfg.ephys_param:
            raise RuntimeError(
                "At least one of `ephys_regex`, `ephys_root`, or `ephys_param` is set, "
                "but not all of them. Please set all three to use ephys recordings.\n"
                f"{cfg.ephys_root=}\n{cfg.ephys_regex=}\n{cfg.ephys_param=}"
            )
        else:
            ephys = None

        return cls(name=cfg.name,
                   root=root,
                   fps=cfg.fps,
                   recordings=recordings,
                   calibrations=calibrations,
                   view_config=cfg.views,
                   keypoints=cfg.keypoints,
                   ephys_recordings=ephys,
                   ephys_param=cfg.ephys_param,
                   sync=cfg.sync)

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
        if self.ephys_param:
            tab.add_row(
                "Ephys Params",
                ", ".join([f"{k}: {v}" for k, v in self.ephys_param.items()])
            )
        else:
            tab.add_row("Ephys Params", "N/A")
        pty.print(tab)
        # print keypoint info
        tab = table.Table("Label", "Group(s)", "View(s)", title="Project keypoints")
        for pt in self.keypoints:
            tab.add_row(pt.label, ", ".join(pt.groups), ", ".join(pt.views))
        pty.print(tab)
        # print recording info
        tab = table.Table("Recording", "Files", title="Project recordings")
        for group, files in self.recordings.items():
            tab.add_row(group.as_str(),
                        ",\n".join([f"{view}: {file.relative_to(self.path)}"
                                    for view, file in files.items()]))
        pty.print(tab)
        # print ephys info
        if self.ephys_param:
            tab = table.Table("Recording", "Files", title="Project ephys recordings")
            for group, file in self.ephys_recordings.items(): # type: ignore
                tab.add_row(group.as_str(), str(file.relative_to(self.path)))
            pty.print(tab)
        # print calibration info
        tab = table.Table("Recording", "Files", title="Project calibrations")
        for group, files in self.calibrations.items():
            tab.add_row(group.as_str(),
                        ",\n".join([f"{view}: {file.relative_to(self.path)}"
                                    for view, file in files.items()]))
        pty.print(tab)

    def synchronize(self):
        # run video synchronization first
        for recording, views in self.recordings.items():
            rprint(f"[bold green]Synchronizing recording videos:[/bold green] {recording.name}")
            ref_video = views[self.sync.ref_view]
            ref_crop = self.view_config[self.sync.ref_view].get_crop(self.sync.ref_crop)
            for view, video in views.items():
                if view == self.sync.ref_view:
                    continue
                crop = self.view_config[view].get_crop(self.sync.ref_crop)
                ref_reader = VideoSyncReader(source=self.path / ref_video,
                                             sample_rate=self.fps,
                                             threshold=self.sync.led_threshold,
                                             crop=ref_crop)
                target_reader = VideoSyncReader(source=self.path / video,
                                                sample_rate=self.fps,
                                                threshold=self.sync.led_threshold,
                                                crop=crop)
                pipeline = SyncPipeline.from_cfg(self.sync, ref_reader, target_reader)
                align_params = pipeline.align_recording()
                pipeline.write_json(align_params)
        # run ephys synchronization if available
        if self.ephys_recordings and self.ephys_param:
            for recording, ephys_file in self.ephys_recordings.items():
                rprint(f"[bold green]Synchronizing recording ephys:[/bold green] {recording.name}")
                ref_video = self.recordings[recording][self.sync.ref_view]
                crop = self.view_config[self.sync.ref_view].get_crop(self.sync.ref_crop)
                video_reader = VideoSyncReader(source=self.path / ref_video,
                                               sample_rate=self.fps,
                                               threshold=self.sync.led_threshold,
                                               crop=crop)
                ephys_reader = get_ephys_reader(self.path / ephys_file, self.ephys_param)
                pipeline = SyncPipeline.from_cfg(self.sync, video_reader, ephys_reader)
                align_params = pipeline.align_recording()
                pipeline.write_json(align_params)
