import itertools
import matplotlib.image as mpimg
import os
import re
from glob import glob
from omegaconf import DictConfig
from dataclasses import dataclass
from typing import Callable, Tuple, Any

def unzip(iter):
    return tuple(list(x) for x in zip(*iter))

def maybe(this, that):
    return that if this is None else this

def bodyparts(cfg: DictConfig):
    return {k: v for k, v in cfg.dataset.bodyparts.items()
                 if (v is not None) and len(v) > 0}

def processed_video_name(video, filters):
    if filters is None:
        return video
    else:
        path, ext = os.path.splitext(video)
        return f"{path}_processed{ext}"

def processed_videos(cfg: DictConfig):
    return [processed_video_name(v, f)
            for v, f in zip(cfg.dataset.videos, cfg.dataset.filters)]

def cropframe(image, crop_coords):
    sx, ex, sy, ey = crop_coords

    return image[sy:ey, sx:ex]

def loadframes(cfg: DictConfig, i):
    videos = processed_videos(cfg)
    paths = [os.path.splitext(v)[0] for v in videos]
    frames = [mpimg.imread(p + f" ({i}).png") for p in paths]

    return frames

def flatten(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def reglob(pattern, path = None, recursive = False):
    """
    `glob` a filesystem using regex patterns.

    Arguments:
    - `pattern`: a regex pattern compatible with Python's `re`
    - `path`: the root under which to search
    """
    path = os.getcwd() if path is None else path
    files = glob(os.sep.join([path, "**"]), recursive=recursive)
    regex = re.compile(pattern)

    return sorted([f for f in files if regex.search(f) is not None])

@dataclass
class Serializable:
    """
    A serializable instance of a class.
    Useful when you are trying to use a class instance with joblib
    but some fields of the class are not serializable.
    As long as the class can be reconstructed from serializable arguments,
    this wrapper provides a way to hold those arguments and constructor.
    You then close over the Serilizable instance and reconstruct within each job.

    Arguments:
    - `constructor`: a callable such that `constructor(*fields)` returns the
        original class instance
    - `fields`: a tuple of arguments to be passed to `constructor`
    """
    constructor: Callable
    fields: Tuple[Any, ...]

    def reconstruct(self):
        return self.constructor(*self.fields)
