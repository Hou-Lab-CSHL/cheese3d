import re
import os
import toml
from pathlib import Path
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from rich import table, console
from rich import print as rprint
from typing import List, Dict, Optional, Any
from collections import namedtuple

from cheese3d.config import (MultiViewConfig,
                             KeypointConfig,
                             ModelConfig,
                             TriangulationConfig,
                             ProjectConfig,
                             keypoints_by_group)
from cheese3d.synchronize.core import SyncConfig, SyncPipeline
from cheese3d.synchronize.readers import VideoSyncReader, get_ephys_reader
from cheese3d.backends.core import Pose2dBackend
from cheese3d.backends.dlc import DLCBackend
from cheese3d.utils import reglob, maybe, get_group_pattern

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

def build_model_backend(cfg: ModelConfig | str | Path,
                        root: Path,
                        recordings: Dict[RecordingKey, Dict[str, Path]],
                        view_cfg: MultiViewConfig,
                        keypoints: List[KeypointConfig]):
    if isinstance(cfg, str) or isinstance(cfg, Path):
        videos = []
        crops = []
        for recording in recordings.values():
            for view, video in recording.items():
                videos.append(video)
                crops.append(view_cfg[view].get_crop())
        existing_project = Path(cfg)
        name = existing_project.name.split("-", 1)[0]
        root = root / name / "backend"

        return DLCBackend.from_existing(existing_project, root, videos, keypoints, crops)
    elif cfg.backend_type == "dlc":
        if cfg.name is None:
            return None

        videos = []
        crops = []
        for recording in recordings.values():
            for view, video in recording.items():
                videos.append(video)
                crops.append(view_cfg[view].get_crop())

        return DLCBackend(
            name=cfg.name,
            root_dir=root / cfg.name / "backend",
            videos=videos,
            keypoints=keypoints,
            experimenter=cfg.backend_options.get("experimenter", "default"),
            date=cfg.backend_options.get("date"),
            crops=crops
        )
    else:
        raise RuntimeError(f"Unrecognized model backend {cfg.backend_type}.")

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
    model_root: str
    fps: int
    recordings: Dict[RecordingKey, Dict[str, Path]]
    calibrations: Dict[RecordingKey, Dict[str, Path]]
    view_config: MultiViewConfig
    view_regex: str
    keypoints: List[KeypointConfig]
    model: Optional[Pose2dBackend]
    ephys_recordings: Optional[Dict[RecordingKey, Path]] = None
    ephys_param: Optional[Dict[str, Any]] = None
    sync: SyncConfig = field(
        default_factory=lambda: SyncConfig(["crosscorr", "regression", "sample_rate"])
    )
    triangulation: TriangulationConfig = field(default_factory=TriangulationConfig)
    ignore_keypoint_labels: List[str] = field(default_factory=list)

    @property
    def path(self):
        return self.root / self.name

    @property
    def model_path(self):
        return self.path / self.model_root

    @property
    def triangulation_path(self):
        return self.path / "triangulation"

    @staticmethod
    def initialize(name: str, root: str | Path, skip_model = False):
        location = Path(root) / name
        if location.exists():
            raise RuntimeError(f"Project {name} already exists under {root}")
        # create project directory
        location.mkdir(parents=True)
        # create a empty configuration file
        cfg = ProjectConfig.default(skip_model=skip_model)
        cfg.name = name
        with location / "config.yaml" as f:
            OmegaConf.save(cfg, f)

    @classmethod
    def from_cfg(cls, cfg: ProjectConfig, root: str | Path, model_import = None):
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
        model_cfg = maybe(model_import, cfg.model)
        model = build_model_backend(model_cfg,
                                    root=(root / cfg.name / cfg.model_root),
                                    recordings=recordings,
                                    view_cfg=cfg.views,
                                    keypoints=cfg.keypoints)
        view_regex = get_group_pattern(ProjectConfig.build_regex(cfg.video_regex), "view")

        return cls(name=cfg.name,
                   root=root,
                   model_root=cfg.model_root,
                   fps=cfg.fps,
                   model=model,
                   recordings=recordings,
                   calibrations=calibrations,
                   view_config=cfg.views,
                   view_regex=view_regex,
                   keypoints=cfg.keypoints,
                   ephys_recordings=ephys,
                   ephys_param=cfg.ephys_param,
                   sync=cfg.sync,
                   triangulation=cfg.triangulation,
                   ignore_keypoint_labels=cfg.ignore_keypoint_labels)

    @classmethod
    def from_path(cls, path: str | Path,
                  cfg_dir = None, overrides = None, model_import = None):
        path = Path(path)
        cfg_file = path / "config.yaml"
        cfg = ProjectConfig.load(cfg_file, cfg_dir, overrides)

        return cls.from_cfg(cfg, path.parent, model_import=model_import) # type: ignore

    def summarize(self):
        pty = console.Console()
        # print basic info
        tab = table.Table(title="Cheese3D project info")
        tab.add_column("Key")
        tab.add_column("Value")
        tab.add_row("Name", self.name)
        tab.add_row("Root Path", str(self.root))
        tab.add_row("Model Path", str(self.model_path))
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

    def _create_labels(self):
        if self.model is None:
            raise RuntimeError("Cannot create labels when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?")
        # create label root if it doesn't exist
        label_path = self.model_path / self.model.name / "labels"
        label_path.mkdir(exist_ok=True)
        # create label folders for each video
        for recording in self.recordings.values():
            for video in recording.values():
                label_folder = label_path / video.stem
                label_folder.mkdir(exist_ok=True)

    def _import_labels(self):
        if self.model is None:
            raise RuntimeError("Cannot import labels when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?")
        self._create_labels()
        label_paths = {
            p.name: p
            for p in map(Path, reglob(r".*", str(self.model_path / self.model.name / "labels")))
        }
        self.model.import_c3d_labels(label_paths)

    def _export_labels(self):
        if self.model is None:
            raise RuntimeError("Cannot export labels when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?")
        self._create_labels()
        label_paths = {
            p.name: p
            for p in map(Path, reglob(r".*", str(self.model_path / self.model.name / "labels")))
        }
        self.model.export_c3d_labels(label_paths)

    def extract_frames(self):
        self._import_labels()
        if self.model is None:
            raise RuntimeError("Cannot extract frames when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?")
        self.model.extract_frames()
        self._export_labels()

    def label_frames(self):
        raise NotImplementedError("Labeling tool not integrated yet.")

    def train(self, gpu):
        self._import_labels()
        if self.model is None:
            raise RuntimeError("Cannot train model when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?")
        self.model.train(gpu)

    def _setup_anipose(self):
        if self.model is None:
            raise RuntimeError("Cannot setup triangulation when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?")
        # make anipose project folder
        self.triangulation_path.mkdir(exist_ok=True)
        # create session subfolders
        for recording, videos in self.recordings.items():
            session_path = self.triangulation_path / recording.name
            session_path.mkdir(exist_ok=True)
            # add raw videos
            videos_path = session_path / "videos-raw"
            videos_path.mkdir(exist_ok=True)
            for video in videos.values():
                src = Path(self.path / video)
                dst = videos_path / src.name
                relpath = Path(os.path.relpath(src, videos_path))
                if dst.exists():
                    os.remove(dst)
                os.symlink(relpath, dst)
            # add calibration
            calibration_path = session_path / "calibration"
            calibration_path.mkdir(exist_ok=True)
            # add calibration files
            cal_key = RecordingKey(recording.session, recording.name)
            matches = [k for k in self.calibrations.keys() if cal_key.matches(k)]
            if len(matches) == 0:
                raise RuntimeError(f"No calibration found for {recording} when setting up triangulation")
            for match in matches:
                for video in self.calibrations[match].values():
                    src = Path(self.path / video)
                    dst = calibration_path / src.name
                    relpath = Path(os.path.relpath(src, calibration_path))
                    if dst.exists():
                        os.remove(dst)
                    os.symlink(relpath, dst)
        # create anipose config file
        kp_schema = keypoints_by_group(self.keypoints)
        for group, kps in kp_schema.items():
            if len(kps) > 2:
                kp_schema[group].append(kps[0])
        config = {
            "project": self.name,
            "model_folder": os.path.relpath(self.model.project_path, self.triangulation_path),
            "nesting": 1,
            "pipeline": {"videos-raw": "videos-raw",},
            "labeling": {
                "scheme": list(kp_schema.values()),
                "ignore": self.ignore_keypoint_labels
            },
            "filter": {
                "enabled": self.triangulation.filter2d,
                "type": "medfilt",
                "medfilt": 13, # length of median filter
                "offset_threshold": 5, # offset from median filter to count as jump
                "score_threshold": 0.8, # score below which to count as bad
                "spline": False, # interpolate using linearly instead of cubic spline
            },
            "calibration": {
                "board_type": "charuco",
                "board_size": [7, 7],
                "board_marker_bits": 4,
                "board_marker_dict_number": 50,
                "board_marker_length": 4.5, # mm
                "board_square_side_length": 6 # mm
            },
            "triangulation": {
                "triangulate": True,
                "cam_regex": f"({self.view_regex})",
                "manually_verify": False,
                "axes": self.triangulation.axes,
                "reference_point": self.triangulation.ref_point,
                "optim": True,
                "score_threshold": self.triangulation.score_threshold,
                "scale_smooth": 0.0,
            }
        }
        with open(self.triangulation_path / "config.toml", "w") as f:
            toml.dump(config, f)

    def _load_anipose_cfg(self):
        from anipose.anipose import load_config
        self._setup_anipose()

        return load_config(str(self.triangulation_path / "config.toml"))

    def calibrate(self):
        from anipose.calibrate import calibrate_all
        calibrate_all(self._load_anipose_cfg())

    def track(self):
        from anipose.pose_videos import pose_videos_all
        pose_videos_all(self._load_anipose_cfg())

    def triangulate(self):
        from anipose.triangulate import triangulate_all
        triangulate_all(self._load_anipose_cfg())
