import cv2
# Must run before matplotlib is imported below: importing cv2 (line above)
# repoints Qt's platform-plugin path at cv2's own bundled Qt build, which
# makes Matplotlib's default `qtagg` backend abort the process (SIGABRT)
# when it initializes Qt -- even for figures that are only ever saved, as
# every figure here is outside `plot_debug`. See the helper's docstring.
from cheese3d.utils import clear_opencv_qt_platform_plugin_override

clear_opencv_qt_platform_plugin_override()

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
from open_ephys.analysis import Session as OESession

import cheese3d.allego_fr as afr
from cheese3d.registry import load_entry_points, registered_names
from cheese3d.utils import maybe, BoundingBox, VideoFrames

SYNC_READER_REGISTRY: Dict[str, Any] = {}

def register_sync_reader(name: str, reader_cls):
    SYNC_READER_REGISTRY[name] = reader_cls

def get_sync_reader_class(name: str):
    if name not in SYNC_READER_REGISTRY:
        load_entry_points("cheese3d.sync_readers", SYNC_READER_REGISTRY)
    if name not in SYNC_READER_REGISTRY:
        raise RuntimeError(f"Unknown sync reader type: {name}. "
                           f"Supported types are: {registered_names(SYNC_READER_REGISTRY)}")

    return SYNC_READER_REGISTRY[name]

@dataclass
class SyncSignalReader:
    """Abstract base class for reading synchronization signals from a source.

    Attributes:
    - `source`: Path to the source file.
    - `sample_rate`: Sample rate of the signal.
    - `threshold`: Threshold for detecting sync events from raw signals
        (specific interpretation depends on subclass).
    - `time_start`: Optional start time for reading the signal.
    - `time_end`: Optional end time for reading the signal.
    """
    source: Path
    sample_rate: int
    threshold: Optional[float] = None
    time_start: Optional[int] = None
    time_end: Optional[int] = None

    def load_signal(self):
        raise NotImplementedError("Attempt to use SyncSignalReader directly. "
                                  "Instead, inherit from this class to implement "
                                  "a specific sync signal source.")

    def root_path(self) -> Path:
        raise NotImplementedError("Attempt to use SyncSignalReader directly. "
                                  "Instead, inherit from this class to implement "
                                  "a specific sync signal source.")

@dataclass
class VideoSyncReader(SyncSignalReader):
    """Reads synchronization signal based on brightness
    within a cropped video file.

    Additional attributes:
    - `crop`: Tuple of (left, right, top, bottom) coordinates for cropping.
    """
    crop: BoundingBox = field(default_factory=lambda: [None, None, None, None])
    peak_algorithm: str = "dynamic"

    def load_signal(self):
        print(self.source)
        video = VideoFrames(str(self.source), bounds=self.crop)
        # get average brightness level
        brightness = []
        for i, frame in enumerate(tqdm(video, desc="process video")):
            if ((self.time_start is not None and i < self.time_start) or
                (self.time_end is not None and i > self.time_end)):
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(np.mean(frame))
        # get peak brightness level
        brightness = np.asarray(brightness)
        hist, bin_edges = np.histogram(brightness, bins=100)
        min_brightness = bin_edges[np.argmax(hist) + 1]
        brightness = brightness - min_brightness
        nz_brightness = brightness[brightness > 2]
        if (self.peak_algorithm == "dynamic") and (len(nz_brightness) > 0):
            mid = np.percentile(nz_brightness, 75)
            # q3 = np.percentile(nz_brightness, 75)
            # iqr = q3 - q1
            # peak_brightness = np.max(nz_brightness[nz_brightness < q3 + 1.5 * iqr])
            peak_brightness = np.percentile(nz_brightness[nz_brightness >= mid], 90)
        else:
            peak_brightness = np.max(brightness)
        # threshold brightness
        led_threshold = maybe(self.threshold, 0.9) * peak_brightness
        led_signal = np.where(brightness > led_threshold, 1, 0)
        # save exemplar frame if possible
        if np.sum(led_signal) > 0:
            exemplar_idx = np.where(led_signal)[0]
            exemplar_frame = video.imgs[exemplar_idx[0]]
            title = f" (frame = {exemplar_idx[0]})"
        else:
            exemplar_frame = video.imgs[0]
            title = " (no match)"

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(brightness)
        ax.axhline(peak_brightness, color='k', linestyle='--')
        ax.axhline(led_threshold, color='r', linestyle='--')
        ax.set_title("LED BBox Brightness")
        fig.savefig(f"{video.path.rstrip('.avi')}-qc-brightness.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        path = Path(video.path).stem
        left, right, top, bottom = video.bounds
        left = maybe(left, 0)
        right = maybe(right, exemplar_frame.shape[1])
        top = maybe(top, 0)
        bottom = maybe(bottom, exemplar_frame.shape[0])
        ax.imshow(exemplar_frame, cmap="gray")
        ax.set_title(f"{path}{title}")
        rect = patches.Rectangle((left, top), right - left, bottom - top,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.axis('off')
        fig.savefig(f"{video.path.rstrip('.avi')}-qc-bbox.png", bbox_inches="tight")
        plt.close(fig)

        return led_signal

    def root_path(self):
        return self.source.parent.joinpath(self.source.stem)

@dataclass
class AllegoSyncReader(SyncSignalReader):
    channel: int = 32

    def load_signal(self):
        tstart = None if self.time_start is None else self.time_start * self.sample_rate
        tend = None if self.time_end is None else self.time_end * self.sample_rate
        channels, _, _ = afr.read_allego_xdat_all_signals(
            datasource_name=self.root_path(),
            time_start=tstart,
            time_end=tend
        )
        analog_signal = channels[self.channel]
        threshold = maybe(self.threshold, 0.1)
        analog_signal = np.where(analog_signal > threshold, 1, 0)

        return analog_signal

    def root_path(self):
        return self.source.parent.joinpath(Path(self.source.stem).stem)

@dataclass
class OpenEphysSyncReader(SyncSignalReader):
    channel: int = 32

    def load_signal(self):
        oerecording = OESession(self.source).recordnodes[1].recordings[0]
        stored_sample_rate = oerecording.info["continuous"][0]["sample_rate"]
        if stored_sample_rate != self.sample_rate:
            logging.warning(f"OpenEphys {stored_sample_rate=} != {self.sample_rate=}")
        tstart = 0 if self.time_start is None else self.time_start * self.sample_rate
        tend = -1 if self.time_end is None else self.time_end * self.sample_rate
        channels = oerecording.continuous[0].get_samples(
            start_sample_index=tstart,
            end_sample_index=tend
        )
        # openephys sync has a short positive on pulse
        # followed by a short negative off pulse
        # we compute the rising (falling) edge of the positive (negative) pulses
        # then we create a binary signal that is high between these edges
        analog_signal = channels[:, self.channel]
        threshold = maybe(self.threshold, 0.1)
        analog_signal_pos = np.where(analog_signal > threshold, 1, 0)
        analog_signal_pos = np.where(np.diff(analog_signal_pos))[0]
        analog_signal_neg = np.where(-analog_signal > threshold, 1, 0)
        analog_signal_neg = np.where(np.diff(analog_signal_neg))[0]
        analog_signal = np.zeros_like(analog_signal)
        for start, end in zip(analog_signal_pos, analog_signal_neg):
            analog_signal[start:end] = 1

        return analog_signal

    def root_path(self):
        return self.source

class DSISyncReader(SyncSignalReader):
    def load_signal(self):
        led_df = pd.read_csv(self.source, sep="\t", names=["timestamp", "signal"])
        if self.time_start is not None:
            time_start = self.time_start
            led_df = led_df.query("timestamp >= @time_start")
        if self.time_end is not None:
            time_end = self.time_end
            led_df = led_df.query("timestamp <= @time_end")
        analog_signal = led_df["signal"].values
        threshold = maybe(self.threshold, 0.1)
        analog_signal = np.where(analog_signal > threshold, 1, 0)

        return analog_signal

    def root_path(self):
        return self.source.parent.joinpath(self.source.stem.removesuffix("_led"))

class SpikeGLXSyncReader(SyncSignalReader):
    def load_signal(self):
        on_file = self.source / "XA0_ON.txt"
        off_file = self.source / "XA0_OFF.txt"
        if not on_file.exists():
            raise FileNotFoundError(f"ON edges file not found: {on_file}")
        if not off_file.exists():
            raise FileNotFoundError(f"OFF edges file not found: {off_file}")
        # Load edge sample indices
        on_edges = np.loadtxt(on_file, dtype=float)
        off_edges = np.loadtxt(off_file, dtype=float)
        # Determine signal length
        if self.time_start is not None and self.time_end is not None:
            analog_tbase = np.arange(self.time_start, self.time_end) / self.sample_rate
        elif self.time_end is not None:
            analog_tbase = np.arange(0, self.time_end) / self.sample_rate
        else:
            # Use the last edge + buffer
            all_edges = np.concatenate([on_edges, off_edges])
            last_timebin = all_edges.max() + 1
            if all_edges.min() > 0:
                first_timebin = 0
            analog_tbase = np.arange(first_timebin, last_timebin, 1/self.sample_rate)
        # Create binary signal
        analog_signal = np.zeros_like(analog_tbase)
        # Use searchsorted for fast index lookup
        on_indices = np.searchsorted(analog_tbase, on_edges)
        off_indices = np.searchsorted(analog_tbase, off_edges)
        # Apply high signal between ON and OFF indices
        for on, off in zip(on_indices, off_indices):
            analog_signal[on:off] = 1

        return analog_signal

    def root_path(self):
        return self.source / self.source.stem

def get_ephys_reader(source: str | Path, ephys_param: Dict[str, Any]):
    reader_cls = get_sync_reader_class(ephys_param["type"])
    kwargs = {k: v for k, v in ephys_param.items()
              if k not in ["type", "sync_threshold", "sync_channel"]}
    kwargs["source"] = Path(source)
    kwargs["threshold"] = ephys_param.get("sync_threshold")
    if "sync_channel" in ephys_param:
        kwargs["channel"] = ephys_param["sync_channel"]

    return reader_cls(**kwargs)

register_sync_reader("allego", AllegoSyncReader)
register_sync_reader("openephys", OpenEphysSyncReader)
register_sync_reader("dsi", DSISyncReader)
