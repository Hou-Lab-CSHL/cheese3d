import os
import re
import pims
import cv2
import sre_parse
from glob import glob
from typing import List, Optional
from contextlib import contextmanager
from pathlib import Path

# (top left x, top left y, bottom right x, bottom right y)
# (xstart, xend, ystart, yend)
BoundingBox = List[Optional[int]]

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
        cv2.destroyAllWindows() # type: ignore

        return height, width

def cropframe(image, crop_coords):
    sx, ex, sy, ey = crop_coords

    return image[sy:ey, sx:ex]

def unzip(iter):
    return tuple(list(x) for x in zip(*iter))

def maybe(this, that):
    return that if this is None else this

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
