import re
import os
import io
import toml
import tempfile
import tarfile
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from rich import table, console
from rich import print as rprint
from typing import List, Dict, Optional, Any
from collections import namedtuple
from datetime import datetime
from filelock import FileLock

from cheese3d.anatomy import compute_anatomical_measurements
from cheese3d.config import (MultiViewConfig,
                             KeypointConfig,
                             KeypointGroupConfig,
                             ModelConfig,
                             TriangulationConfig,
                             ProjectConfig,
                             keypoints_by_group,
                             _DEFAULT_KEYPOINTS)
from cheese3d.synchronize.core import SyncConfig, synchronize_videos, synchronize_ephys
from cheese3d.backends.core import Pose2dBackend, get_pose_backend_class
from cheese3d.utils import (dlc_folder_to_components,
                            read_3d_data,
                            reglob,
                            maybe,
                            get_group_pattern,
                            relative_path,
                            is_subpath)

class RecordingKey(namedtuple("RecordingKey", ["session", "name", "attributes"])):
    __slots__ = () # prevent __dict__ creation since subclassing namedtuple

    def __new__(cls, session: str, name: str, **attributes):
        return super().__new__(cls, session, name, frozenset(attributes.items()))

    def __eq__(self, other):
        if isinstance(other, RecordingKey):
            return (self.name == other.name) and self.matches(other)
        else:
            return False

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
                sessions: List[Dict[str, str]],
                views: MultiViewConfig):
    videos = {}
    calibration_videos = {}
    for recording in sessions:
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
                if (match is None) or (match.group("view") != video_cfg.view):
                    continue
                if all(match.group(k) == v
                       for k, v in recording.items() if k != "name"):
                    view_start, view_end = match.span("view")
                    group_name = match.group(0)
                    group_name = Path(group_name[:view_start] + group_name[view_end:]).stem
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

def find_ephys(dir: Path, ephys_regex: str, sessions: Dict[RecordingKey, Dict[str, Path]]):
    ephys = {}
    grouped_sessions = group_by_session(sessions)

    for session, session_sessions in grouped_sessions.items():
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
        for recording in session_sessions.keys():
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
                        sessions: Dict[RecordingKey, Dict[str, Path]],
                        view_cfg: MultiViewConfig,
                        keypoints: List[KeypointConfig],
                        keypoint_groups=None):
    if isinstance(cfg, str) or isinstance(cfg, Path):
        videos = []
        crops = []
        for recording in sessions.values():
            for view, video in recording.items():
                videos.append(video)
                crops.append(view_cfg[view].get_crop())
        existing_project = Path(cfg)
        name, *_ = dlc_folder_to_components(existing_project)
        root = root / name / "backend"
        backend_cls = get_pose_backend_class("dlc")

        return backend_cls.from_existing(
            existing_project, root, videos, keypoints, crops,
            skeleton=[edge for group in (keypoint_groups or []) for edge in group.skeleton],
        )
    else:
        if cfg.name is None:
            return None

        videos = []
        crops = []
        for recording in sessions.values():
            for view, video in recording.items():
                videos.append(video)
                crops.append(view_cfg[view].get_crop())
        if cfg.backend_type == "eks" and "view_names" not in cfg.backend_options:
            cfg.backend_options["view_names"] = {
                view: view_cfg[view].view
                for view in view_cfg
            }
        if cfg.backend_type == "eks" and "camera_names" not in cfg.backend_options:
            cfg.backend_options["camera_names"] = [
                view for view in cfg.backend_options["view_names"].values()
            ]
        backend_cls = get_pose_backend_class(cfg.backend_type)

        backend_options = dict(cfg.backend_options)
        if cfg.backend_type == "dlc":
            # DLC's skeleton was formerly left as its two placeholder edges.
            # Flatten Cheese3D's anatomical groups for DLC visualization and PAFs.
            backend_options["skeleton"] = [
                edge for group in (keypoint_groups or []) for edge in group.skeleton
            ]

        return backend_cls(name=cfg.name,
                           root_dir=root / cfg.name / "backend",
                           videos=videos,
                           keypoints=keypoints,
                           crops=crops,
                           **backend_options)

def resolve_pose3d_csv(session_path: str | Path) -> Path:
    """Return the single Anipose 3-D pose CSV produced for a session.

    Anipose normally appends the pose scorer/model name to the recording stem,
    so its output cannot be reconstructed reliably from the recording name.
    """
    pose3d_dir = Path(session_path) / "pose-3d"
    csv_files = sorted(pose3d_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No 3-D pose CSV found in {pose3d_dir}. "
            "Run triangulation before opening the visualizer."
        )
    if len(csv_files) > 1:
        choices = "\n".join(f"  - {path.name}" for path in csv_files)
        raise RuntimeError(
            f"Multiple 3-D pose CSV files found in {pose3d_dir}; cannot determine "
            f"which result to visualize:\n{choices}"
        )

    return csv_files[0]

@dataclass
class Ch3DProject:
    """
    A Cheese3D project.

    Arguments:
        - `name`: the name of the project
        - `root`: root directory under which the project folder should be made
        - `sessions`: a list of sessions where each entry is video files
            organized by camera view
        - `keypoints`: a list of `KeypointConfig`s to track in this project
    """
    name: str
    root: Path
    video_root: Path
    model_root: Path
    fps: int
    sessions: Dict[RecordingKey, Dict[str, Path]]
    calibrations: Dict[RecordingKey, Dict[str, Path]]
    view_config: MultiViewConfig
    view_regex: str
    keypoints: List[KeypointConfig]
    keypoint_groups: List[KeypointGroupConfig]
    model: Optional[Pose2dBackend]
    ephys_root: Optional[Path] = None
    ephys_sessions: Optional[Dict[RecordingKey, Path]] = None
    ephys_param: Optional[Dict[str, Any]] = None
    sync: SyncConfig = field(
        default_factory=lambda: SyncConfig(["crosscorr", "regression", "sample_rate"])
    )
    triangulation: TriangulationConfig = field(default_factory=TriangulationConfig)
    ignore_keypoint_labels: List[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return self.root / self.name

    @property
    def model_path(self) -> Path:
        return self.path / relative_path(self.model_root, self.path)

    @property
    def recording_path(self) -> Path:
        return self.path / relative_path(self.video_root, self.path)

    @property
    def ephys_path(self) -> Optional[Path]:
        if self.ephys_root:
            return self.path / relative_path(self.ephys_root, self.path)
        else:
            return None

    @property
    def triangulation_path(self) -> Path:
        return self.path / "triangulation"

    @property
    def checkpoint_path(self) -> Path:
        return self.path / "checkpoints"

    @staticmethod
    def initialize(name: str, root: str | Path, skip_model = False):
        location = Path(root) / name
        if (location / "config.yaml").exists():
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
        sessions, calibrations = find_videos(
            dir=root / cfg.name / relative_path(cfg.video_root, root / cfg.name),
            recording_regex=ProjectConfig.build_regex(cfg.video_regex),
            calibration_keys=cfg.calibration,
            sessions=cfg.sessions,
            views=cfg.views
        )
        if cfg.ephys_regex and cfg.ephys_root and cfg.ephys_param:
            ephys = find_ephys(
                dir=root / cfg.name / relative_path(cfg.ephys_root, root / cfg.name),
                ephys_regex=ProjectConfig.build_regex(cfg.ephys_regex),
                sessions=sessions
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
        with FileLock(root / cfg.name / "build_backend.lock"):
            model = build_model_backend(model_cfg,
                                        root=(root / cfg.name /
                                            relative_path(cfg.model_root, root / cfg.name)),
                                        sessions=sessions,
                                        view_cfg=cfg.views,
                                        keypoints=cfg.keypoints,
                                        keypoint_groups=cfg.keypoint_groups)
        view_regex = get_group_pattern(ProjectConfig.build_regex(cfg.video_regex), "view")

        return cls(name=cfg.name,
                   root=root,
                   video_root=Path(cfg.video_root),
                   ephys_root=Path(cfg.ephys_root) if cfg.ephys_root else None,
                   model_root=Path(cfg.model_root),
                   fps=cfg.fps,
                   model=model,
                   sessions=sessions,
                   calibrations=calibrations,
                   view_config=cfg.views,
                   view_regex=view_regex,
                   keypoints=cfg.keypoints,
                   keypoint_groups=cfg.keypoint_groups,
                   ephys_sessions=ephys,
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

        return cls.from_cfg(cfg, path.parent, model_import=model_import)

    def summarize(self, pty = None):
        pty = maybe(pty, console.Console())
        # print basic info
        tab = table.Table(title="Cheese3D project info")
        tab.add_column("Key")
        tab.add_column("Value")
        tab.add_row("Name", self.name)
        tab.add_row("Root Path", str(self.root))
        tab.add_row("Video Path", str(self.recording_path))
        tab.add_row("Model Path", str(self.model_path))
        if self.ephys_param:
            tab.add_row("Ephys Path", str(self.ephys_path))
            tab.add_row(
                "Ephys Params",
                ", ".join([f"{k}: {v}" for k, v in self.ephys_param.items()])
            )
        else:
            tab.add_row("Ephys Path", "N/A")
            tab.add_row("Ephys Params", "N/A")
        pty.print(tab)
        # print keypoint info
        tab = table.Table("Label", "Group(s)", "View(s)", title="Project keypoints")
        for pt in self.keypoints:
            tab.add_row(pt.label, ", ".join(pt.groups), ", ".join(pt.views))
        pty.print(tab)
        # print recording info
        tab = table.Table("Session", "Files (relative to Video Path)", title="Project sessions")
        for group, files in self.sessions.items():
            tab.add_row(group.as_str(),
                        ",\n".join([f"{view}: {file.relative_to(self.recording_path)}"
                                    for view, file in files.items()]))
        pty.print(tab)
        # print ephys info
        if self.ephys_param:
            tab = table.Table("Session", "Files (relative to Ephys Path)", title="Project ephys sessions")
            for group, file in self.ephys_sessions.items(): # type: ignore
                tab.add_row(group.as_str(), str(file.relative_to(self.recording_path)))
            pty.print(tab)
        # print calibration info
        tab = table.Table("Session", "Files (relative to Recording Path)", title="Project calibrations")
        for group, files in self.calibrations.items():
            tab.add_row(group.as_str(),
                        ",\n".join([f"{view}: {file.relative_to(self.recording_path)}"
                                    for view, file in files.items()]))
        pty.print(tab)

    def synchronize(self):
        # run video synchronization first
        for recording, views in self.sessions.items():
            rprint(f"[bold green]Synchronizing recording videos:[/bold green] {recording.name}")
            ref_video = views[self.sync.ref_view]
            ref_crop = self.view_config[self.sync.ref_view].get_crop(self.sync.ref_crop)
            sync_targets = {}
            for view, video in views.items():
                if view == self.sync.ref_view:
                    continue
                crop = self.view_config[view].get_crop(self.sync.ref_crop)
                sync_targets[view] = (video, crop)
            synchronize_videos(self.sync, (ref_video, ref_crop), sync_targets, fps=self.fps)
        # run ephys synchronization if available
        if self.ephys_sessions and self.ephys_param:
            for recording, ephys_file in self.ephys_sessions.items():
                rprint(f"[bold green]Synchronizing recording ephys:[/bold green] {recording.name}")
                ref_video = self.sessions[recording][self.sync.ref_view]
                ref_crop = self.view_config[self.sync.ref_view].get_crop(self.sync.ref_crop)
                ephys_path = self.path / ephys_file
                synchronize_ephys(self.sync, (ref_video, ref_crop), ephys_path, self.ephys_param, fps=self.fps)

    def _create_labels(self):
        if self.model is None:
            raise RuntimeError("Cannot create labels when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?")
        # create label root if it doesn't exist
        label_path = self.model_path / self.model.name / "labels"
        label_path.mkdir(parents=True, exist_ok=True)
        # create label folders for each video
        for recording in self.sessions.values():
            for video in recording.values():
                label_folder = label_path / video.stem
                label_folder.mkdir(parents=True, exist_ok=True)

    def _label_folder_paths(self):
        if self.model is None:
            raise RuntimeError("Cannot find labels when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?")

        return {
            p.name: p
            for p in map(Path, reglob(r".*", str(self.model_path / self.model.name / "labels")))
        }

    def _import_labels(self):
        if self.model is None:
            raise RuntimeError("Cannot import labels when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?")
        with FileLock(self.path / "labels.lock"):
            self._create_labels()
            label_paths = self._label_folder_paths()
            self.model.import_c3d_labels(label_paths)

    def _export_labels(self):
        if self.model is None:
            raise RuntimeError("Cannot export labels when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?")
        with FileLock(self.path / "labels.lock"):
            self._create_labels()
            label_paths = self._label_folder_paths()
            self.model.export_c3d_labels(label_paths)

    def extract_frames(self, sessions: Optional[List[RecordingKey]] = None, manual = False):
        self._import_labels()
        if self.model is None:
            raise RuntimeError("Cannot extract frames when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?")

        if manual:
            if not sessions:
                raise ValueError("A list of sessions must be specified in manual extraction mode.")

            import napari
            from cheese3d_annotator.widget import FramePickerWidget

            for recording in sessions:
                rprint(f"Extracting {recording.name} ... close Napari window when complete.")
                viewer = napari.Viewer()
                picker = FramePickerWidget(viewer)
                viewer.window.add_dock_widget(picker)
                picker.set_videos([v for v in self.sessions[recording].values()])
                picker.set_save_directory(self.model_path / self.model.name / "labels")
                viewer.show(block=True)
        else:
            self.model.extract_frames()
        self._export_labels()

    def label_frames(self):
        if self.model is None:
            raise RuntimeError("Cannot label frames when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?")

        import napari
        from cheese3d_annotator.widget import FrameAnnotatorWidget

        self._export_labels()
        viewer = napari.Viewer()
        annotator = FrameAnnotatorWidget(viewer)
        viewer.window.add_dock_widget(annotator)
        annotator.set_file_dialogs(img_folder=(self.model_path / self.model.name / "labels"),
                                   config_file=(self.path / "config.yaml"))
        viewer.show(block=True)
        self._import_labels()

    def train(self, gpu, iterate_dataset=True, training_settings=None):
        """Train the selected backend with optional GUI/CLI settings."""
        self._import_labels()
        if self.model is None:
            raise RuntimeError("Cannot train model when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?")
        self.model.train(gpu, iterate_dataset, training_settings)

    def _setup_anipose(self):
        if self.model is None:
            raise RuntimeError("Cannot setup triangulation when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?")
        with FileLock(self.path / "setup_anipose.lock"):
            # make anipose project folder
            self.triangulation_path.mkdir(exist_ok=True)
            # create session subfolders
            for recording, videos in self.sessions.items():
                session_path = self.triangulation_path / recording.name
                session_path.mkdir(exist_ok=True)
                # add raw videos
                videos_path = session_path / "videos-raw"
                videos_path.mkdir(exist_ok=True)
                for video in videos.values():
                    src = Path(video).resolve()
                    dst = videos_path / src.name
                    relpath = Path(os.path.relpath(src, videos_path.resolve()))
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
                        src = Path(video).resolve()
                        dst = calibration_path / src.name
                        relpath = Path(os.path.relpath(src, calibration_path.resolve()))
                        if dst.exists():
                            os.remove(dst)
                        os.symlink(relpath, dst)
            # create anipose config file
            kp_schema = keypoints_by_group(self.keypoints)
            for group, kps in kp_schema.items():
                if len(kps) > 2:
                    kp_schema[group].append(kps[0])
            model_path = getattr(self.model, "anipose_model_path", self.model.project_path)
            config = {
                "project": self.name,
                "model_folder": os.path.relpath(model_path, os.getcwd()),
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

    def _resolve_anipose_session(self, session: str) -> str:
        session_path = self.triangulation_path / session
        if not session_path.is_dir():
            available = [p.name for p in self.triangulation_path.iterdir()
                         if p.is_dir()] if self.triangulation_path.is_dir() else []
            raise ValueError(
                f"Session folder '{session}' not found in {self.triangulation_path}. "
                f"Available sessions: {available}")
        return str(session_path.resolve())

    def calibrate(self, session: Optional[str] = None):
        if session is not None:
            from anipose.calibrate import process_session
            config = self._load_anipose_cfg()
            process_session(config, self._resolve_anipose_session(session))
        else:
            from anipose.calibrate import calibrate_all
            calibrate_all(self._load_anipose_cfg())

    def track(self, session: Optional[str] = None, tracking_settings=None):
        """Track selected sessions with explicit inference resource settings."""
        if self.model is None:
            raise RuntimeError("Cannot track when pose model does not exist "
                               "(hint: maybe you forgot to set `model.name` in the config?)")
        self._setup_anipose()
        recordings = [(recording, videos) for recording, videos in self.sessions.items()
                      if session is None or recording.name == session]
        if len(recordings) == 0:
            raise ValueError(f"No recordings matched session={session!r}.")
        for recording, videos in recordings:
            rprint(f"[bold green]Tracking {recording.name} with {len(videos)} videos...[/bold green]")
            output_dir = self.triangulation_path / recording.name / "pose-2d"
            calibration_path = (self.triangulation_path / recording.name /
                                "calibration" / "calibration.toml")
            handled = self.model.track(videos=videos,
                                       output_dir=output_dir,
                                       calibration_path=calibration_path,
                                       tracking_settings=tracking_settings)
            if not handled:
                from anipose.pose_videos import process_session
                rprint("[bold yellow]WARNING:[/bold yellow] Pose backend did not handle "
                       "tracking directly; falling back to Anipose tracking.")
                process_session(self._load_anipose_cfg(),
                                self._resolve_anipose_session(recording.name))

    def triangulate(self, session: Optional[str] = None):
        # first triangulate points using Anipose
        if session is not None:
            from anipose.triangulate import process_session
            config = self._load_anipose_cfg()
            process_session(config, self._resolve_anipose_session(session))
            sessions_to_process = [self.triangulation_path / session]
        else:
            from anipose.triangulate import triangulate_all
            triangulate_all(self._load_anipose_cfg())
            sessions_to_process = [s for s in self.triangulation_path.iterdir()
                                   if s.is_dir()]
        # now compute cheese3d features
        exclude_kps = set(kp.label for kp in _DEFAULT_KEYPOINTS) - set(kp.label for kp in self.keypoints)
        if len(exclude_kps) > 0:
            rprint("[bold red]Keypoint configuration does not match default Cheese3D keypoints. "
                   "Some Cheese3D features may not be computed![/bold red]")
        rprint("Generating Cheese3d features ...")
        for session_dir in sessions_to_process:
            if session_dir.is_dir():
                landmarks = read_3d_data(session_dir)
                if landmarks is None:
                    rprint(f"[bold red]No landmarks found for {session_dir.name}, skipping![/bold red]")
                    continue
                c3d_features = compute_anatomical_measurements(landmarks, exclude_kps)
                # write features to csv
                c3d_features_df = pd.DataFrame({
                    k: v
                    for _, features in c3d_features.items()
                    for k, v in features.items()
                })
                if len(c3d_features_df) == 0:
                    rprint(f"[bold red]No features constructed for {session_dir.name}, skipping![/bold red]")
                    continue
                csv_output = (session_dir / "cheese3d")
                csv_output.mkdir(exist_ok=True)
                csv_output = csv_output / "cheese3d_features.csv"
                if not csv_output.exists():
                    c3d_features_df.to_csv(csv_output, index=False, index_label=None)

    def generate_videos(self, max_workers: Optional[int] = None):
        """Generate QC videos using adjustable camera-level CPU parallelism."""
        from anipose.project_2d import process_session as project_2d_session
        from cheese3d.generate_videos import generate_videos_2d
        from cheese3d.generate_videos import generate_compare_video

        self._setup_anipose()
        kp_schema = keypoints_by_group(self.keypoints)
        for group, kps in kp_schema.items():
            if len(kps) > 2:
                kp_schema[group].append(kps[0])
        scheme = list(kp_schema.values())
        bodyparts = sorted(set([bp for chain in scheme for bp in chain]))
        completed = 0
        for recording, _ in self.sessions.items():
            session_path = self.triangulation_path / recording.name
            videos_raw_dir = session_path / "videos-raw"
            pose_2d_dir = session_path / "pose-2d"
            pose_2d_filt_dir = session_path / "pose-2d-filtered"
            pose_3d_dir = session_path / "pose-3d"
            calib_dir = session_path / "calibration"
            pose_2d_proj_dir = session_path / "pose-2d-proj"
            videos_labeled_dir = session_path / "videos-labeled"
            videos_labeled_filt_dir = session_path / "videos-labeled-filtered"
            videos_2d_proj_dir = session_path / "videos-2d-proj"
            videos_compare_dir = session_path / "videos-compare"

            rprint(f"[bold]Labeling videos in 2D: {recording.name}[/bold]")
            completed += generate_videos_2d(
                scheme, bodyparts, videos_raw_dir, pose_2d_dir, videos_labeled_dir,
                max_workers=max_workers,
            )
            if self.triangulation.filter2d:
                rprint(f"[bold]Labeling filtered videos in 2D: {recording.name}[/bold]")
                completed += generate_videos_2d(scheme,
                                                bodyparts,
                                                videos_raw_dir,
                                                pose_2d_filt_dir,
                                                videos_labeled_filt_dir,
                                                max_workers=max_workers)
            calib_toml = calib_dir / "calibration.toml"
            pose_3d_csvs = sorted(pose_3d_dir.glob("*.csv"))
            if len(pose_3d_csvs) > 0 and calib_toml.exists():
                ap_cfg = self._load_anipose_cfg()
                # Detect video extension from actual files for anipose
                raw_files = [f for f in videos_raw_dir.iterdir() if f.is_file()]
                if raw_files:
                    ap_cfg['video_extension'] = raw_files[0].suffix.lstrip('.')
                rprint(f"[bold]Projecting 3D points to 2D: {recording.name}[/bold]")
                project_2d_session(ap_cfg, str(session_path.resolve()))
                rprint(f"[bold]Labeling reprojected 3D points in 2D: {recording.name}[/bold]")
                completed += generate_videos_2d(scheme,
                                                bodyparts,
                                                videos_raw_dir,
                                                pose_2d_proj_dir,
                                                videos_2d_proj_dir,
                                                max_workers=max_workers)
            if videos_labeled_dir.exists() and videos_2d_proj_dir.exists():
                rprint(f"[bold]Stitching labeled videos together: {recording.name}[/bold]")
                completed += generate_compare_video(videos_raw_dir,
                                                    videos_labeled_dir,
                                                    videos_labeled_filt_dir,
                                                    videos_2d_proj_dir,
                                                    videos_compare_dir,
                                                    f"({self.view_regex})")

        if completed == 0:
            raise RuntimeError(
                "Video generation produced no outputs. Check the log for missing pose files "
                "or unmatched raw videos."
            )

        return completed

    def visualize(self, recording: RecordingKey):
        import napari
        from cheese3d_annotator.data_visualizer.qc_video import QCReprojApp
        from cheese3d_annotator.data_visualizer.rig_view import RigViewer
        from cheese3d_annotator.data_visualizer.widget import _SyncController

        videos = {recording.name: {
            self.view_config[view].view: str(path)
            for view, path in self.sessions[recording].items()
        }}
        anipose_folder = self.triangulation_path / recording.name
        calibration = anipose_folder / "calibration" / "calibration.toml"
        pose3d = resolve_pose3d_csv(anipose_folder)
        c3d_features = anipose_folder / "cheese3d" / "cheese3d_features.csv"
        view_names = {cfg.view: view for view, cfg in self.view_config.items()}
        skeleton = []
        for group in self.keypoint_groups:
            skeleton.extend(group.skeleton)
        # Preserve the configured long view names because QC must not judge a
        # keypoint in cameras that were excluded from its triangulation.
        keypoint_views = {kp.label: list(kp.views) for kp in self.keypoints}
        qc = QCReprojApp(videos_by_group=videos,
                         calibration_path=calibration,
                         pose3d_csv=pose3d,
                         view_code_to_name=view_names,
                         skeleton_config=skeleton,
                         keypoint_views=keypoint_views,
                         pose2d_dir=anipose_folder / "pose-2d")
        rig = RigViewer(calibration_path=calibration,
                        features_csv=c3d_features,
                        annotation_path=pose3d,
                        skeleton_config=skeleton)
        _SyncController(rig, qc)
        napari.run()

    def checkpoint(self, skip_source = False, portable = False):
        rprint("Creating archive of project (this make take several minutes) ... ")
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.checkpoint_path.mkdir(exist_ok=True)
        with tarfile.open(self.checkpoint_path / f"{self.name}_{timestamp}.tar.xz", "x:xz") as tar:
            # add project folder, skipping potential relative paths
            def _filter(x: tarfile.TarInfo):
                if (x.name.startswith(str(self.recording_path)) or
                    x.name.startswith(str(self.model_path)) or
                    (self.ephys_path and x.name.startswith(str(self.ephys_path))) or
                    x.name.startswith(str(self.checkpoint_path)) or
                    x.name.startswith(str(self.path / "config.yaml"))):
                    return None
                else:
                    return x
            tar.add(self.path, arcname=self.name, filter=_filter)
            config = OmegaConf.load(self.path / "config.yaml")
            def _resolve_symlinks(x: tarfile.TarInfo):
                if x.islnk():
                    # Create new TarInfo from real file
                    full_path = Path(x.name).resolve()
                    new_info = tarfile.TarInfo(name=x.name)
                    new_info.size = os.path.getsize(full_path)
                    new_info.mode = os.stat(full_path).st_mode

                    return new_info
                else:
                    return x
            def _replace_root(x: tarfile.TarInfo, rel_root: Path):
                # x.name now is relative like "project/path/to/file.txt"
                # We want it to be like "project/videos/session1/file.txt"
                name_parts = Path(x.name).parts
                if name_parts and name_parts[0] == self.name:
                    # Strip project prefix and rebuild relative path
                    filename = name_parts[-1]  # Just the filename
                    x.name = str(rel_root / filename)
                return x
            if not skip_source and is_subpath(self.recording_path, start=self.path):
                if portable:
                    tar.add(self.recording_path, arcname=f"{self.name}/{self.video_root}",
                            filter=_resolve_symlinks)
                else:
                    tar.add(self.recording_path, arcname=f"{self.name}/{self.video_root}")
            elif not skip_source and portable:
                config["video_root"] = "videos" # type: ignore
                for session in self.sessions.values():
                    for recording in session.values():
                        rel_root = Path(self.name) / "videos" / recording.parent.name
                        tar.add(recording, filter=(lambda x: _replace_root(x, rel_root)))
                for session in self.calibrations.values():
                    for recording in session.values():
                        rel_root = Path(self.name) / "videos" / recording.parent.name
                        tar.add(recording, filter=(lambda x: _replace_root(x, rel_root)))
            if is_subpath(self.model_path, start=self.path):
                tar.add(self.model_path, arcname=f"{self.name}/{self.model_root}")
            elif portable:
                config["model_root"] = "models" # type: ignore
                for path in self.model_path.iterdir():
                    rel_root = Path(self.name) / "models"
                    tar.add(path, filter=(lambda x: _replace_root(x, rel_root)))
            if self.ephys_path:
                ephys_root_rel = self.ephys_root.name if isinstance(self.ephys_root, Path) else self.ephys_root
                if not skip_source and is_subpath(self.ephys_path, start=self.path):
                    if portable:
                        tar.add(self.ephys_path, arcname=f"{self.name}/{ephys_root_rel}",
                                filter=_resolve_symlinks)
                    else:
                        tar.add(self.ephys_path, arcname=f"{self.name}/{ephys_root_rel}")
                elif not skip_source and portable:
                    config["ephys_root"] = "ephys" # type: ignore
                    for recording in self.ephys_sessions.values(): # type: ignore
                        rel_root = Path(self.name) / "ephys" / recording.parent.name
                        tar.add(recording, filter=(lambda x: _replace_root(x, rel_root)))
            yaml_str = OmegaConf.to_yaml(config)
            yaml_bytes = yaml_str.encode("utf-8")
            arcname_config = f"{self.name}/config.yaml"
            tarinfo = tarfile.TarInfo(arcname_config)
            tarinfo.size = len(yaml_bytes)
            tar.addfile(tarinfo, fileobj=io.BytesIO(yaml_bytes))

    def restore(self, checkpoint_path, skip_source = False):
        rprint("Restoring project from checkpoint (this may take several minutes)...")
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        # Read config from checkpoint first to get video_root/ephys_root values
        video_root_chk = None
        ephys_root_chk = None
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.yaml', delete=False) as tmp_config:
            tmp_config_path = Path(tmp_config.name)
        try:
            with tarfile.open(checkpoint_file, "r:xz") as tar:
                config_member = None
                for member in tar.getmembers():
                    if member.name.endswith("config.yaml"):
                        config_member = member
                        break
                if config_member:
                    config_data = tar.extractfile(config_member)
                    if config_data:
                        tmp_config_path.write_bytes(config_data.read())
                        config = ProjectConfig.load(tmp_config_path, cfg_dir=None, overrides=None)
                        video_root_chk = config.video_root
                        ephys_root_chk = config.ephys_root
        finally:
            if tmp_config_path.exists():
                tmp_config_path.unlink()
        def should_extract(name):
            if name.endswith("config.yaml"):
                return True
            if name.endswith("/"):
                return True
            path_parts = Path(name).parts
            if "checkpoints" in path_parts:
                return False
            if skip_source:
                if video_root_chk and (video_root_chk in path_parts):
                    return False
                if ephys_root_chk and (ephys_root_chk in path_parts):
                    return False
            return True
        def extract_filter(members):
            for member in members:
                if should_extract(member.name):
                    yield member
        # Create project directory if it doesn't exist
        self.path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(checkpoint_file, "r:xz") as tar:
            for member in extract_filter(tar.getmembers()):
                tar.extract(member, path=self.root)
