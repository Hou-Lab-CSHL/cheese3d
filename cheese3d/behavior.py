import pims
import cv2
from contextlib import contextmanager

from cheese3d.utils import cropframe, Serializable

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
        cap = cv2.VideoCapture(self.path)
        try:
            yield cap
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def __iter__(self):
        # use opencv for faster iteration
        sx, ex, sy, ey = self.bounds
        with self.opencv_capture() as video:
            video.set(cv2.CAP_PROP_POS_FRAMES, max(self.shift, 0))
            for _ in range(len(self)):
                ret, frame = video.read()
                if ret:
                    yield frame[sy:ey, sx:ex]
                else:
                    break

    def __str__(self):
        return str(self.path)

    def serialize(self):
        return Serializable(VideoFrames, (self.path, self.shift, self.bounds))

class MulticamView(dict):
    """A mult-camera recording view. This class behaves like a `dict` of views.

    Different views can be accessed as if they are attributes
    (e.g. `myviews.topleft` is equivalent to `myviews.views["topleft"]`).
    view name.
    """
    def __getattr__(self, name):
        if name in self.keys():
            return self[name]
        else:
            raise AttributeError(name=name, obj=self)

    def as_dict(self):
        return dict(self)

    def as_list(self):
        return list(self.values())
