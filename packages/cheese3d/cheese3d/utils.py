import os
import re
import pims
import cv2
import numpy as np
import pandas as pd
from glob import glob
from typing import List, Optional, Tuple
from contextlib import contextmanager
from pathlib import Path

# (top left x, top left y, bottom right x, bottom right y)
# (xstart, xend, ystart, yend)
BoundingBox = List[Optional[int]]
RGB = Tuple[int, int, int]

class VideoFrames:
    """
    A (potentially time shifted) video indexed by frames for a recording session
    cropped to a bounding box region.

    Arguments:
    - `path`: a path to the video file
    - `shift = 0`: the +/- shift in units of video frames
    - `bounds = [None, None, None, None]`: a tuple of the form
        `[xstart, xend, ystart, yend]` (set any element to `None` to use the
        max bounds)
    """
    def __init__(self, path, shift = 0, bounds = [None, None, None, None]):
        self.imgs = pims.Video(path)
        self.shift = shift
        self.bounds = bounds
        self.path = path

    def shifted_index(self, i):
        return max(i + self.shift, 0)

    def __getitem__(self, i):
        return cropframe(self.imgs[self.shifted_index(i)], self.bounds)

    def __len__(self):
        return len(self.imgs) - self.shift

    @contextmanager
    def opencv_capture(self):
        cap = cv2.VideoCapture(self.path) # type: ignore
        try:
            yield cap
        finally:
            cap.release()
            cv2.destroyAllWindows() # type: ignore

    def __iter__(self):
        # use opencv for faster iteration
        sx, ex, sy, ey = self.bounds
        with self.opencv_capture() as video:
            video.set(cv2.CAP_PROP_POS_FRAMES, max(self.shift, 0)) # type: ignore
            for _ in range(len(self)):
                ret, frame = video.read()
                if ret:
                    yield frame[sy:ey, sx:ex]
                else:
                    break

    def __str__(self):
        return str(self.path)

    @staticmethod
    def get_dims(path: str | Path):
        cap = cv2.VideoCapture(str(path)) # type: ignore
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # type: ignore
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # type: ignore
        cap.release()
        # cv2.destroyAllWindows() # type: ignore

        return height, width

def cropframe(image, crop_coords):
    sx, ex, sy, ey = crop_coords

    return image[sy:ey, sx:ex]

def unzip(iter):
    return tuple(list(x) for x in zip(*iter))

def maybe(this, that):
    return that if this is None else this

def relative_path(path: str | Path, start: str | Path):
    path = Path(path)
    if path.is_absolute():
        return os.path.relpath(path, start)
    else:
        return path

def reglob(pattern, path = None, recursive = False):
    """
    `glob` a filesystem using regex patterns.

    Arguments:
    - `pattern`: a regex pattern compatible with Python's `re`
    - `path`: the root under which to search
    - `recursive`: set to true to search the path recursively
    """
    path = os.getcwd() if path is None else path
    files = glob(os.sep.join([path, "**"]), recursive=recursive)
    regex = re.compile(pattern)

    return sorted([f for f in files if regex.search(f) is not None])

def dlc_folder_to_components(folder: str | Path):
    *name, experimenter, year, month, date = Path(folder).name.split("-")

    return "-".join(name), experimenter, "-".join([year, month, date])

def read_3d_data(data_dir: str | Path, extra_cols = None):
    """Load 3D landmark data from Anipose.

    ### Arguments
    - `data_dir`: the Anipose directory for the video to load
    - `extra_cols`: optional, a list of extra columns to include in the output
       where each extra column is a string appended to a landmark name
       (e.g. "error" in this list becomes "lowerlip_error" for the 3D CSV header)

    ### Returns
    A dictionary of 3D data where each key is a landmark name
    and the associated value is an array of shape `(time, 3)`.
    If `extra_cols` is provided, an additional dictionary is returned where each
    entry corresponds to an extra column and the value is dictionary of landmarks
    to arrays of shape `(time,)`.
    """
    files = glob(str(Path(data_dir) / "pose-3d" / "*.csv"))
    data = pd.read_csv(files[0])
    cols = data.head() # name of all the columns
    landmark_names = np.unique([s.split('_')[0]
                                for s in cols if s.endswith(('_x', '_y', '_z'))])
    landmarks = {landmark: np.stack([data[f"{landmark}_x"],
                                     data[f"{landmark}_y"],
                                     data[f"{landmark}_z"]], axis=-1)
                 for landmark in landmark_names}

    if extra_cols is not None:
        extra_landmarks = {col: {landmark: np.asarray(data[f"{landmark}_{col}"])
                                 for landmark in landmark_names}
                           for col in extra_cols}

        return landmarks, extra_landmarks
    else:
        return landmarks

def tiny_cmap(center, delta, n):
    import seaborn as sns
    import matplotlib as mpl

    if isinstance(center, str):
        _cmap = sns.color_palette(center, as_cmap=True)
    elif isinstance(center, (list, tuple)):
        _cmap = sns.dark_palette(tuple(center), input="rgb", as_cmap=True)

    if isinstance(_cmap, list):
        _cmap = mpl.colors.LinearSegmentedColormap.from_list("tiny_cmap", _cmap)

    return list(_cmap(np.linspace(center - delta, center + delta, n)))

def get_group_pattern(regex: str, group_name: str):
    """
    Return the raw regex subpattern text for the named group `group_name`, if present.
    Supports common syntaxes:
        - (?P<name>...)
        - (?<name>...)
        - (?'name'...)
    Handles nesting, escapes, character classes, and (?#...) comments.
    Returns None if the named group is not found or is malformed.
    """

    def find_matching_paren(s: str, open_idx: int) -> Optional[int]:
        n = len(s)
        i = open_idx + 1
        depth = 1
        in_class = False
        escaped = False
        while i < n:
            ch = s[i]
            if escaped:
                escaped = False
                i += 1
                continue
            if ch == '\\':
                escaped = True
                i += 1
                continue
            if in_class:
                if ch == ']':
                    in_class = False
                i += 1
                continue
            if ch == '[':
                in_class = True
                i += 1
                continue
            if ch == '(':
                # Comment group: (?#...)
                if i + 2 < n and s[i+1] == '?' and s[i+2] == '#':
                    # Count this '(' then skip to its closing ')'
                    depth += 1
                    i += 3
                    while i < n:
                        if s[i] == '\\':
                            i += 2
                            continue
                        if s[i] == ')':
                            depth -= 1
                            i += 1
                            break
                        i += 1
                    continue
                depth += 1
                i += 1
                continue
            if ch == ')':
                depth -= 1
                if depth == 0:
                    return i
                i += 1
                continue
            i += 1
        return None

    s = regex
    n = len(s)
    i = 0
    in_class = False
    escaped = False
    while i < n:
        ch = s[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == '\\':
            escaped = True
            i += 1
            continue
        if in_class:
            if ch == ']':
                in_class = False
            i += 1
            continue
        if ch == '[':
            in_class = True
            i += 1
            continue
        if ch == '(':
            if i + 1 < n and s[i+1] == '?':
                # (?P<name>...)
                tok = f"?P<{group_name}>"
                if s.startswith(tok, i + 1):
                    start = i + 1 + len(tok)
                    end = find_matching_paren(s, i)
                    return s[start:end] if end is not None else None # type: ignore
                # (?<name>...)
                tok = f"?<{group_name}>"
                if s.startswith(tok, i + 1):
                    start = i + 1 + len(tok)
                    end = find_matching_paren(s, i)
                    return s[start:end] if end is not None else None # type: ignore
                # (?'name'...)
                tok = f"?'{group_name}'"
                if s.startswith(tok, i + 1):
                    start = i + 1 + len(tok)
                    end = find_matching_paren(s, i)
                    return s[start:end] if end is not None else None # type: ignore
            i += 1
            continue
        i += 1

    return None
