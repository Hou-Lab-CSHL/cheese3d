
import hydra
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from collections import defaultdict
from omegaconf import OmegaConf

from cheese3d.behavior import MulticamView, VideoFrames
from cheese3d.ffmpeg import filter_videos
from cheese3d.utils import maybe

BoundingBox = List[Optional[int]]

@dataclass
class VideoConfig:
    """
    Structured config specification for a single camera recording.

    Arguments:
    - `path`: path to the video file(s) for this camera
    - `crop`: a bounding box of the form `(xstart, xend, ystart, yend)`
        where any coordinate maybe set to `None` to accept the index bounds
    - `extra_crops`: a dictionary of bounding boxes structured similar to `crop`
    - `filterspec`: a filter specs for FFMPEG specified as a dictionary
        of the form (set to `None` for no filter):
        - `brightness`: value in [-1.0, 1.0] (default 0)
        - `contrast`: value in [-1000.0, 1000.0] (default 1)
        - `saturation`: value in [0.0, 3.0] (default 1)
    """
    path: Any
    crop: BoundingBox = (None, None, None, None)
    extra_crops: Optional[Dict[str, BoundingBox]] = None
    filterspec: Optional[Dict[str, List[float]]] = None

    def get_crop(self, crop = "default"):
        if crop == "default":
            return self.crop
        elif self.extra_crops is None:
            return (None, None, None, None)
        else:
            return self.extra_crops.get(crop, (None, None, None, None))

    def instantiate(self, shift = 0, crop = "default"):
        """
        Return a `fepipeline.behavior.VideoFrames` instance based on this config.

        Arguments:
        - `shift`: an optional +/- shift (in frames) applied to the video data
        - `crop`: set to `"default"` to crop video using `self.crop` or set to
            a key in `self.extra_crops` to use that bounding box instead
        """
        if isinstance(self.path, str):
            video = filter_videos([self.path], [self.filterspec])[0]
            crop_area = self.get_crop(crop)

            return VideoFrames(video, shift, crop_area)
        else:
            videos = filter_videos(self.path, [self.filterspec for _ in self.path])
            crop_area = self.get_crop(crop)

            return [VideoFrames(video, shift, crop_area) for video in videos]

    def as_list(self):
        if isinstance(self.path, str):
            return [self.path]
        else:
            return self.path

class MultiViewConfig(dict):
    """Structured config specification for multi-camera setups.
    This behaves like dictionary of `VideoConfig`s.

    Different views can be accessed as if they are attributes
    (e.g. `myviews.topleft` is equivalent to `myviews["topleft"]`).
    view name.

    Arguments:
    - `views`: a dictionary of `VideoConfig`s with keys corresponding
    to the view name and
    """
    def __getattr__(self, name):
        if name in self.keys():
            return self[name]
        else:
            raise AttributeError(name=name, obj=self)

    def instantiate(self, shifts = None, crops = None):
        shifts = maybe(shifts, defaultdict(lambda: 0))
        crops = maybe(crops, defaultdict(lambda: "default"))

        return MulticamView({view: cfg.instantiate(shifts[view], crops[view])
                             for view, cfg in self.items()})

    def as_dict(self):
        return dict(self)

    def as_list(self):
        return list(self.values())

@dataclass
class MetadataConfig:
    """
    Structured config specification for recording metadata stored as a CSV.

    Arguments:
    - `path`: path to the CSV file
    - `headers`: a list of headers for each column of the CSV
    """
    path: str
    headers: List[str] = ("LTimestamp", "LIndex", "RTimestamp", "RIndex", "Event")

    def read(self):
        dtypes = {"Event": str} if "Event" in self.headers else {}

        return pd.read_csv(self.path, names=self.headers, dtype=dtypes) # type: ignore

@dataclass
class SessionConfig:
    """
    Structured config specification for a behavioral session.

    Arguments:
    - `name`: the name of the session
    - `mouse`: the mouse in the recording
    - `condition`: the condition (variant) to use
    - `run`: the run number to use
    - `cal_run`: the run number to use for calibration videos
    - `videos`: an instance of a `AbstractViewConfig`
    - `calibration`: an instance of a `AbstractViewConfig` for calibration videos
    - `metadata`: a `MetadataConfig`
    - `fps`: frames per second for the video data
    - `stimulus_window`: a dictionary w/ the number of seconds `before` and `after`
        each lick that is marks the start and end of a trial
    - `mistrials`: a list of trials that should be ignored from analysis
    - `bodyparts`: a dictionary of landmarks to extract grouped by cluster, e.g.:

        ```python
        {"eye": ["eye(back)", "eye(front)", "eye(bottom)", eye(top)"],
         "nose": ["nose(tip)", nose(bottom)", "nose(top)"]}
        ```
    - `cam_bodyparts`: a dictionary of landmarks grouped by camera view
    """
    name: str
    mouse: Optional[str]
    condition: Optional[str]
    run: Optional[str]
    cal_run: Optional[str]
    concat_videos: bool
    videos: Dict[str, VideoConfig]
    calibration: Optional[Dict[str, VideoConfig]]
    metadata: MetadataConfig
    ephys: Optional[Any]
    fps: int
    stimulus_window: Dict[str, int]
    mistrials: List[int]
    bodyparts: Dict[str, List[str]]
    cam_bodyparts: Dict[str, List[str]]

    def __post_init__(self):
        self.videos = MultiViewConfig(self.videos)
        self.calibration = MultiViewConfig(self.calibration) # type: ignore

def load_sessions(sessions, default_config = "hou-rig2", overrides = None):
    session_cfgs = []
    for session in sessions:
        base_config = session.get("config", default_config)
        _overrides = maybe(overrides, [])
        _overrides.append(f"dataset.name={session['name']}")
        if "mouse" in session:
            _overrides.append(f"dataset.mouse={session['mouse']}")
        if "condition" in session:
            _overrides.append(f"dataset.condition={session['condition']}")
        if "run" in session:
            _overrides.append(f"dataset.run={session['run']}")
        if "cal_run" in session:
            _overrides.append(f"dataset.cal_run={session['cal_run']}")
        if "concat_videos" in session:
            _overrides.append(f"dataset.concat_videos={session['concat_videos']}")

        common_cfg = hydra.compose(config_name="common",
                                   overrides=["+dataset={}"],
                                   return_hydra_config=True)
        cfg = hydra.compose(config_name=f"dataset/{base_config}",
                            overrides=_overrides,
                            return_hydra_config=False)
        cfg = OmegaConf.merge(common_cfg, cfg)
        session_cfgs.append(OmegaConf.to_object(cfg.dataset))

    return session_cfgs
