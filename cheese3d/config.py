import os
import hydra
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path

from cheese3d.utils import maybe

# (top left x, top left y, bottom right x, bottom right y)
# (xstart, xend, ystart, yend)
BoundingBox = List[Optional[int]]

@dataclass
class VideoConfig:
    """
    Structured config specification for a single camera recording.

    Arguments:
    - `path`: a regex string that identifies videos recorded by this camera from filenames
    - `crop`: a bounding box of the form `(xstart, xend, ystart, yend)`
        where any coordinate maybe set to `None` to accept the full bounds
    - `extra_crops`: a dictionary of bounding boxes structured similar to `crop`
    - `filterspec`: a filter specs for FFMPEG specified as a dictionary
        of the form (set to `None` for no filter):
        - `brightness`: value in [-1.0, 1.0] (default 0)
        - `contrast`: value in [-1000.0, 1000.0] (default 1)
        - `saturation`: value in [0.0, 3.0] (default 1)
    """
    path: str = MISSING
    crop: BoundingBox = field(default_factory=lambda: [None, None, None, None])
    extra_crops: Optional[Dict[str, BoundingBox]] = None
    filterspec: Optional[Dict[str, float]] = None

    def get_crop(self, crop = "default"):
        if crop == "default":
            return self.crop
        elif self.extra_crops is None:
            return (None, None, None, None)
        else:
            return self.extra_crops.get(crop, (None, None, None, None))

    # def instantiate(self, shift = 0, crop = "default"):
    #     """
    #     Return a `fepipeline.behavior.VideoFrames` instance based on this config.

    #     Arguments:
    #     - `shift`: an optional +/- shift (in frames) applied to the video data
    #     - `crop`: set to `"default"` to crop video using `self.crop` or set to
    #         a key in `self.extra_crops` to use that bounding box instead
    #     """
    #     if isinstance(self.path, str):
    #         video = filter_videos([self.path], [self.filterspec])[0]
    #         crop_area = self.get_crop(crop)

    #         return VideoFrames(video, shift, crop_area)
    #     else:
    #         videos = filter_videos(self.path, [self.filterspec for _ in self.path])
    #         crop_area = self.get_crop(crop)

    #         return [VideoFrames(video, shift, crop_area) for video in videos]

    def as_list(self):
        if isinstance(self.path, str):
            return [self.path]
        else:
            return self.path

class MultiViewConfig(dict[str, VideoConfig]):
    """
    Structured config specification for multi-camera setups.
    This behaves like dictionary of `VideoConfig`s.

    Different views can be accessed as if they are attributes
    (e.g. `myviews.topleft` is equivalent to `myviews["topleft"]`).

    Arguments:
    - `views`: a dictionary of `VideoConfig`s with keys corresponding
        to the view name.
    """
    def __getattr__(self, name):
        if name in self.keys():
            return self[name]
        else:
            raise AttributeError(name=name, obj=self)

    def __setattr__(self, name: str, value: VideoConfig, /) -> None:
        self[name] = value

    # def instantiate(self, shifts = None, crops = None):
    #     shifts = maybe(shifts, defaultdict(lambda: 0))
    #     crops = maybe(crops, defaultdict(lambda: "default"))

    #     return MulticamView({view: cfg.instantiate(shifts[view], crops[view])
    #                          for view, cfg in self.items()})

    def as_dict(self):
        return dict(self)

    def as_list(self):
        return list(self.values())

class SixCamViewConfig(MultiViewConfig):
    def __init__(self):
        super().__init__()
        self.topleft = VideoConfig(".*_TL_.*.avi")
        self.topright = VideoConfig(".*_TR_.*.avi")
        self.left = VideoConfig(".*_L_.*.avi")
        self.right = VideoConfig(".*_R_.*.avi")
        self.topcenter = VideoConfig(".*_TC_.*.avi")
        self.bottomcenter = VideoConfig(".*_BC_.*.avi")

@dataclass
class KeypointConfig:
    label: str = MISSING
    groups: List[str] = field(default_factory=(lambda: ["default"]))
    views: List[str] = field(default_factory=(lambda: []))

_DEFAULT_KEYPOINTS = [
    KeypointConfig(label="nose(bottom)",
                   groups=["nose"],
                   views=["left", "right", "bottomcenter"]),
    KeypointConfig(label="nose(tip)",
                   groups=["nose"],
                   views=["topleft",
                          "topright",
                          "left",
                          "right",
                          "topcenter",
                          "bottomcenter"]),
    KeypointConfig(label="nose(top)",
                   groups=["nose"],
                   views=["topleft",
                          "topright",
                          "left",
                          "right",
                          "topcenter",
                          "bottomcenter"]),
    KeypointConfig(label="pad(top)(left)",
                   groups=["whiskers(left)"],
                   views=["topleft", "left", "bottomcenter"]),
    KeypointConfig(label="pad(side)(left)",
                   groups=["whiskers(left)"],
                   views=["left", "bottomcenter"]),
    KeypointConfig(label="pad(top)(right)",
                   groups=["whiskers(right)"],
                   views=["topright", "right", "bottomcenter"]),
    KeypointConfig(label="pad(side)(right)",
                   groups=["whiskers(right)"],
                   views=["right", "bottomcenter"]),
    KeypointConfig(label="pad(center)",
                   groups=["whiskers(left)", "whiskers(right)"],
                   views=["left", "right", "bottomcenter"]),
    KeypointConfig(label="lowerlip",
                   groups=["mouth"],
                   views=["left", "right", "bottomcenter"]),
    KeypointConfig(label="upperlip(left)",
                   groups=["mouth"],
                   views=["left", "right", "bottomcenter"]),
    KeypointConfig(label="upperlip(right)",
                   groups=["mouth"],
                   views=["left", "right", "bottomcenter"]),
    KeypointConfig(label="eye(front)(left)",
                   groups=["eye(left)"],
                   views=["topleft", "left", "topcenter"]),
    KeypointConfig(label="eye(top)(left)",
                   groups=["eye(left)"],
                   views=["topleft", "left", "topcenter"]),
    KeypointConfig(label="eye(back)(left)",
                   groups=["eye(left)"],
                   views=["topleft", "left", "topcenter"]),
    KeypointConfig(label="eye(bottom)(left)",
                   groups=["eye(left)"],
                   views=["topleft", "left", "topcenter"]),
    KeypointConfig(label="eye(front)(right)",
                   groups=["eye(right)"],
                   views=["topright", "right", "topcenter"]),
    KeypointConfig(label="eye(top)(right)",
                   groups=["eye(right)"],
                   views=["topright", "right", "topcenter"]),
    KeypointConfig(label="eye(back)(right)",
                   groups=["eye(right)"],
                   views=["topright", "right", "topcenter"]),
    KeypointConfig(label="eye(bottom)(right)",
                   groups=["eye(right)"],
                   views=["topright", "right", "topcenter"]),
    KeypointConfig(label="ear(base)(left)",
                   groups=["ear(left)"],
                   views=["topleft", "left"]),
    KeypointConfig(label="ear(top)(left)",
                   groups=["ear(left)"],
                   views=["topleft", "left"]),
    KeypointConfig(label="ear(tip)(left)",
                   groups=["ear(left)"],
                   views=["topleft", "left"]),
    KeypointConfig(label="ear(bottom)(left)",
                   groups=["ear(left)"],
                   views=["topleft", "left"]),
    KeypointConfig(label="ear(base)(right)",
                   groups=["ear(right)"],
                   views=["topright", "right"]),
    KeypointConfig(label="ear(top)(right)",
                   groups=["ear(right)"],
                   views=["topright", "right"]),
    KeypointConfig(label="ear(tip)(right)",
                   groups=["ear(right)"],
                   views=["topright", "right"]),
    KeypointConfig(label="ear(bottom)(right)",
                   groups=["ear(right)"],
                   views=["topright", "right"]),
    KeypointConfig(label="ref(head-post)",
                   groups=["ref"],
                   views=["topleft",
                          "topright",
                          "left",
                          "right",
                          "topcenter",
                          "bottomcenter"])
]

@dataclass
class ProjectConfig:
    name: str = MISSING
    root: str = os.getcwd()
    recording_root: str = "videos"
    videos: MultiViewConfig = field(default_factory=SixCamViewConfig)
    calibration: str = ".*_cal_.*"
    recording_groups: Optional[Dict[str, str]] = None
    recordings: List[str] = field(default_factory=lambda: [])
    keypoints: List[KeypointConfig] = field(default_factory=lambda: _DEFAULT_KEYPOINTS)

    @classmethod
    def load(cls, cfg_file: str | Path,
             cfg_dir: Optional[str | Path] = None,
             overrides: Optional[List[str]] = None):
        overrides = maybe(overrides, [])
        cfg_file = Path(cfg_file)
        if cfg_dir is not None:
            cfg_dir = Path(cfg_dir)
            overrides.append(f"++hydra.searchpath=[file://{str(cfg_dir.absolute())}]")
        with hydra.initialize_config_dir(str(cfg_file.parent.absolute()),
                                         version_base=None):
            cfg = hydra.compose(cfg_file.name, overrides=overrides)
        schema = OmegaConf.structured(cls)
        cfg = OmegaConf.merge(schema, cfg)

        return cfg
