import hydra
from omegaconf import MISSING, OmegaConf, DictConfig
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path

from cheese3d.utils import maybe, BoundingBox
from cheese3d.synchronize.core import SyncConfig

@dataclass
class VideoConfig:
    """
    Structured config specification for a single camera recording.

    Arguments:
    - `path`: a regex string that identifies videos recorded by this camera from filenames
    - `crop`: a bounding box of the form `(xstart, xend, ystart, yend)`
        where any coordinate maybe set to `None` to accept the full bounds
    - `extra_crops`: a dictionary of bounding boxes structured similar to `crop`
    """
    view: str = MISSING
    crop: BoundingBox = field(default_factory=lambda: [None, None, None, None])
    extra_crops: Optional[Dict[str, BoundingBox]] = None

    def get_crop(self, crop = "default") -> BoundingBox:
        if crop == "default":
            return self.crop
        elif self.extra_crops is None:
            return [None, None, None, None]
        else:
            return self.extra_crops.get(crop, [None, None, None, None])

    def as_list(self):
        if isinstance(self.view, str):
            return [self.view]
        else:
            return self.view

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

    def as_dict(self):
        return dict(self)

    def as_list(self):
        return list(self.values())

class SixCamViewConfig(MultiViewConfig):
    def __init__(self):
        super().__init__()
        self.topleft = VideoConfig("TL")
        self.topright = VideoConfig("TR")
        self.left = VideoConfig("L")
        self.right = VideoConfig("R")
        self.topcenter = VideoConfig("TC")
        self.bottomcenter = VideoConfig("BC")

@dataclass
class KeypointConfig:
    label: str = MISSING
    groups: List[str] = field(default_factory=(lambda: ["default"]))
    views: List[str] = field(default_factory=(lambda: []))

def keypoints_by_group(keypoints: List[KeypointConfig]):
    kp_by_group = {}
    for kp in keypoints:
        for group in kp.groups:
            if group in kp_by_group:
                kp_by_group[group].append(kp.label)
            else:
                kp_by_group[group] = [kp.label]

    return kp_by_group

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
class ModelConfig:
    name: Optional[str] = None
    backend_type: str = "dlc"
    backend_options: Dict[str, Any] = field(default_factory=(lambda: {}))

_DEFAULT_VIDEO_REGEX = {
    "_path_": r".*_{{type}}_{{view}}.*\.avi",
    "type": r"[^_]+",
    "view": r"TL|TR|L|R|TC|BC"
}

@dataclass
class TriangulationConfig:
    axes: List[List[str]] = MISSING
    ref_point: str = MISSING
    filter2d: bool = False
    score_threshold: float = 0.9

_DEFAULT_TRIANGULATION_AXES = [
    [ "z", "nose(top)", "nose(bottom)",],
    [ "x", "eye(front)(left)", "eye(front)(right)",],
]
_DEFAULT_TRIANGULATION_REF = "ref(head-post)"

@dataclass
class ProjectConfig:
    name: str = MISSING
    recording_root: str = "videos"
    ephys_root: Optional[str] = None
    model_root: str = "model"
    video_regex: Any = MISSING
    model: ModelConfig = MISSING
    ephys_regex: Optional[Any] = None
    ephys_param: Optional[Dict[str, Any]] = None
    fps: int = 100
    sync: SyncConfig = MISSING
    recordings: List[Dict[str, str]] = MISSING
    triangulation: TriangulationConfig = MISSING
    views: MultiViewConfig = MISSING
    calibration: Dict[str, str] = MISSING
    keypoints: List[KeypointConfig] = MISSING
    ignore_keypoint_labels: List[str] = MISSING

    @classmethod
    def default(cls, skip_model = False):
        cfg = OmegaConf.structured(cls)
        cfg.video_regex = _DEFAULT_VIDEO_REGEX
        cfg.views = SixCamViewConfig()
        cfg.calibration = {"type": "cal"}
        cfg.recordings = []
        cfg.triangulation = TriangulationConfig(axes=_DEFAULT_TRIANGULATION_AXES,
                                                ref_point=_DEFAULT_TRIANGULATION_REF)
        cfg.keypoints = _DEFAULT_KEYPOINTS
        cfg.ignore_keypoint_labels = ["ref(head-post)"]
        cfg.sync = SyncConfig(["crosscorr", "regression", "samplerate"])
        if not skip_model:
            cfg.model = ModelConfig()

        return cfg

    @staticmethod
    def build_regex(regex):
        if isinstance(regex, str):
            return regex
        elif isinstance(regex, dict) or isinstance(regex, DictConfig):
            if "_path_" in regex:
                full_regex = regex["_path_"]
            else:
                raise RuntimeError("Regex must contain '_path_' key.")
            if "view" not in regex:
                raise RuntimeError("Regex must contain 'view' key.")
            for key, rstr in regex.items():
                full_regex = full_regex.replace("{{" + key + "}}", # type: ignore
                                                fr"(?P<{key}>{rstr})")

            return full_regex
        else:
            raise TypeError("Regex must be a string or a dictionary,"
                            f" got {type(regex)} instead.")

    @classmethod
    def load(cls, cfg_file: str | Path,
             cfg_dir: Optional[str | Path] = None,
             overrides: Optional[List[str]] = None):
        overrides = maybe(overrides, [])
        cfg_file = Path(cfg_file)
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file at path {cfg_file} does not exist.")
        if cfg_dir is not None:
            cfg_dir = Path(cfg_dir)
            overrides.append(f"++hydra.searchpath=[file://{str(cfg_dir.absolute())}]")
        with hydra.initialize_config_dir(str(cfg_file.parent.absolute()),
                                         version_base=None):
            cfg = hydra.compose(cfg_file.name, overrides=overrides)
        schema = OmegaConf.structured(cls)
        cfg = OmegaConf.merge(schema, cfg)
        cfg = OmegaConf.to_object(cfg)

        return cfg
