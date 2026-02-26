import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from pathlib import Path

from cheese3d.synchronize.aligners import (BaseAligner,
                                           CrossCorrelationAligner,
                                           RegressionAligner,
                                           SampleRateAligner,
                                           AlignmentParams)
from cheese3d.synchronize.readers import SyncSignalReader, VideoSyncReader, get_ephys_reader
from cheese3d.synchronize.utils import get_time_points
from cheese3d.utils import maybe, BoundingBox, downsample_video_data

@dataclass
class SyncConfig:
    pipeline: List[str]
    led_threshold: float = 0.9
    max_regression_rmse: float = 1e-2
    ref_view: str = "bottomcenter"
    ref_crop: str = "default"

    def build_pipeline(self, ref_sample_rate, target_sample_rate):
        pipeline = []
        for stage in self.pipeline:
            if stage == "crosscorr":
                pipeline.append(CrossCorrelationAligner(
                    ref_sample_rate=ref_sample_rate,
                    target_sample_rate=target_sample_rate
                ))
            elif stage == "regression":
                pipeline.append(RegressionAligner(
                    ref_sample_rate=ref_sample_rate,
                    target_sample_rate=target_sample_rate,
                    max_rmse=self.max_regression_rmse
                ))
            elif stage == "samplerate":
                pipeline.append(SampleRateAligner(
                    ref_sample_rate=ref_sample_rate,
                    target_sample_rate=target_sample_rate
                ))
            else:
                raise ValueError(f"Unknown alignment stage: {stage}"
                                 " (only 'crosscorr' / 'regression' / 'samplerate' allowed)")

        return pipeline

@dataclass
class SyncPipeline:
    ref: SyncSignalReader
    target: SyncSignalReader
    aligners: List[BaseAligner]

    @classmethod
    def from_cfg(cls, cfg: SyncConfig, ref: SyncSignalReader, target: SyncSignalReader):
        aligners = cfg.build_pipeline(ref.sample_rate, target.sample_rate)

        return cls(ref, target, aligners)

    def find_segment_indices(self, time_signal, segment_type='first'):
        ones_indices = np.where(time_signal)[0]
        if len(ones_indices) == 0:
            return None, None
        if segment_type == 'first':
            start_index = ones_indices[0]
            end_index = np.where(time_signal[start_index:] == 0)[0]
            if len(end_index) == 0:
                end_index = len(time_signal - 1)
            else:
                end_index = end_index[0] + start_index
        elif segment_type == 'last':
            end_index = ones_indices[-1]
            start_index = np.where(time_signal[:end_index] == 0)[0]
            if len(start_index) == 0:
                start_index = 0
            else:
                start_index = start_index[-1]
        elif segment_type == "mid":
            segments = np.split(ones_indices, np.where(np.diff(ones_indices) > 1)[0] + 1)
            middle_segment = segments[len(segments) // 2]
            start_index = middle_segment[0]
            end_index = np.where(time_signal[start_index:] == 0)[0]
            end_index = end_index[0] + start_index
        else:
            raise ValueError("segment_type must be either 'first', 'last', or 'mid'")

        return start_index, end_index

    def plot_alignment(self, ref_signal, target_signal, align_params: AlignmentParams):
        target_sample_rate = maybe(align_params.sample_rate, self.target.sample_rate)
        lag = maybe(align_params.lag, 0)
        ref_time = np.arange(len(ref_signal)) / self.ref.sample_rate
        target_time = np.arange(len(target_signal)) / self.target.sample_rate
        target_time_corrected = (np.arange(len(target_signal)) /
                                 target_sample_rate + lag)
        plt.subplot(4, 1, 1)
        plt.plot(ref_time, ref_signal, label="Ref Signal")
        plt.plot(target_time, target_signal, label="Target Signal")
        plt.ylabel("Signal")
        plt.title("Target Signal and Ref Signal before alignment")
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(ref_time, ref_signal, label="Ref Signal")
        plt.plot(target_time_corrected, target_signal, label="Target Signal")
        plt.xlabel("Time (sec)")
        plt.ylabel("Signal")
        plt.title("Target Signal and Ref Signal after alignment")
        plt.legend()

        plt.subplot(4, 2, 5)
        s, e = self.find_segment_indices(ref_signal, "first")
        extra = (e - s) // 2 + 2
        plt.plot(ref_time[(s - extra):(e + extra)],
                 ref_signal[(s - extra):(e + extra)])
        s, e = self.find_segment_indices(target_signal, "first")
        extra = (e - s) // 2 + 2
        plt.plot(target_time[(s - extra):(e + extra)],
                 target_signal[(s - extra):(e + extra)])
        plt.title("First pulse (before)")
        plt.ylabel("Signal")

        plt.subplot(4, 2, 6)
        s, e = self.find_segment_indices(ref_signal, "last")
        extra = (e - s) // 2 + 2
        plt.plot(ref_time[(s - extra):(e + extra)],
                 ref_signal[(s - extra):(e + extra)])
        s, e = self.find_segment_indices(target_signal, "last")
        extra = (e - s) // 2 + 2
        plt.plot(target_time[(s - extra):(e + extra)],
                 target_signal[(s - extra):(e + extra)])
        plt.title("Last pulse (before)")

        plt.subplot(4, 2, 7)
        s, e = self.find_segment_indices(ref_signal, "first")
        extra = (e - s) // 2 + 2
        plt.plot(ref_time[(s - extra):(e + extra)],
                 ref_signal[(s - extra):(e + extra)])
        s, e = self.find_segment_indices(target_signal, "first")
        extra = (e - s) // 2 + 2
        plt.plot(target_time_corrected[(s - extra):(e + extra)],
                 target_signal[(s - extra):(e + extra)])
        plt.title("First pulse (after)")
        plt.ylabel("Signal")
        plt.xlabel("Time (sec)")

        plt.subplot(4, 2, 8)
        s, e = self.find_segment_indices(ref_signal, "last")
        extra = (e - s) // 2 + 2
        plt.plot(ref_time[(s - extra):(e + extra)],
                 ref_signal[(s - extra):(e + extra)])
        s, e = self.find_segment_indices(target_signal, "last")
        extra = (e - s) // 2 + 2
        plt.plot(target_time_corrected[(s - extra):(e + extra)],
                 target_signal[(s - extra):(e + extra)])
        plt.title("Last pulse (after)")
        plt.xlabel("Time (sec)")

        plt.subplots_adjust(hspace=0.8)
        plt.tight_layout()
        fig = plt.gcf()

        return fig

    def align_recording(self, plot_debug = False):
        ref_signal = self.ref.load_signal()
        target_signal = self.target.load_signal()
        align_params = AlignmentParams(sample_rate=self.target.sample_rate)

        ref_times = get_time_points(ref_signal)
        target_times = get_time_points(target_signal)
        if (len(ref_times) == 0) or (len(target_times) == 0):
            logging.warning(f"No signals detected: {len(ref_times)=}, {len(target_times)=}")

            return align_params
        else:
            print("total ref 'spikes': ", len(ref_times))
            print("total target 'spikes': ", len(target_times))

        for i, aligner in enumerate(self.aligners):
            _align_params, fig = aligner.align(ref_signal, target_signal, align_params)
            align_params = _align_params if _align_params.good else align_params
            if (fig is not None) and plot_debug:
                root = self.target.root_path()
                path = f"{root}.qc-stage-{i}.png"
                fig.savefig(path, bbox_inches="tight")
                plt.show()
            elif fig is not None:
                root = self.target.root_path()
                path = f"{root}.qc-stage-{i}.png"
                fig.savefig(path, bbox_inches="tight")
                plt.close(fig)

        fig = self.plot_alignment(ref_signal, target_signal, align_params)
        fig.savefig(f"{self.target.root_path()}.qc-final.png", bbox_inches="tight")
        if plot_debug:
            plt.show()
        else:
            plt.close(fig)

        return align_params

    def write_json(self, align_params: AlignmentParams):
        if align_params.lag is None:
            return

        time_start = maybe(self.target.time_start, "null")
        time_end = maybe(self.target.time_end, "null")
        slope = maybe(align_params.slope, "null")
        sample_rate = maybe(align_params.sample_rate, self.target.sample_rate)
        params_json = {
            "reference": str(self.ref.source),
            "target": str(self.target.source),
            "lag_time": align_params.lag,
            "slope": slope,
            "sample_rate": sample_rate,
            "time_start" : time_start,
            "time_end" : time_end
        }

        aligner_path = f"{self.target.root_path()}.align.json"
        with open(aligner_path, "w") as outfile:
            json.dump(params_json, outfile)

def synchronize_videos(pipeline_cfg: SyncConfig,
                       reference: Tuple[Path, BoundingBox],
                       videos: Dict[str, Tuple[Path, BoundingBox]],
                       fps: int = 100):
    # run video synchronization first
    ref_video, ref_crop = reference
    ref_reader = VideoSyncReader(source=ref_video,
                                 sample_rate=fps,
                                 threshold=pipeline_cfg.led_threshold,
                                 crop=ref_crop)
    align_params = {"ref": (ref_video, 0, fps)}
    for view, (video, crop) in videos.items():
        target_reader = VideoSyncReader(source=video,
                                        sample_rate=fps,
                                        threshold=pipeline_cfg.led_threshold,
                                        crop=crop)
        pipeline = SyncPipeline.from_cfg(pipeline_cfg, ref_reader, target_reader)
        target_path = pipeline.target.root_path()
        if target_path.with_suffix(".align.json").exists():
            print(f"alignment file exists, skipping {str(target_path)}...")
        else:
            align_param = pipeline.align_recording()
            pipeline.write_json(align_param)
            frame_lag = int(round(maybe(align_param.lag, 0), 2) * align_param.sample_rate)
            align_params[view] = (video, frame_lag, align_param.sample_rate)
    ref_lag = min(-lag for _, lag, _ in align_params.values())
    for video, lag, view_fps in align_params.values():
        shift = -lag - ref_lag
        if view_fps != fps:
            downsample_video_data([str(video)], start_offset=shift, fps=fps)
        else:
            downsample_video_data([str(video)], start_offset=shift)

def synchronize_ephys(pipeline_cfg: SyncConfig,
                      reference: Tuple[Path, BoundingBox],
                      ephys_path: Path,
                      ephys_params: Dict[str, Any],
                      fps: int = 100):
    ref_video, ref_crop = reference
    video_reader = VideoSyncReader(source=ref_video,
                                   sample_rate=fps,
                                   threshold=pipeline_cfg.led_threshold,
                                   crop=ref_crop)
    ephys_reader = get_ephys_reader(ephys_path, ephys_params)
    pipeline = SyncPipeline.from_cfg(pipeline_cfg, video_reader, ephys_reader)
    align_params = pipeline.align_recording()
    pipeline.write_json(align_params)
