import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from typing import Optional, List

from cheese3d.synchronize.utils import get_time_points, resample_signal
from cheese3d.utils import maybe

@dataclass(frozen=True)
class AlignmentParams:
    lag: Optional[float] = None
    slope: Optional[float] = None
    sample_rate: Optional[float] = None
    good: Optional[bool] = None

@dataclass
class BaseAligner:
    """Abstract base class for different alignment protocols.
    Subclasses should implement the `align` method.

    Attributes:
    - `ref_sample_rate`: Sample rate of the reference signal.
    - `target_sample_rate`: Sample rate of the target signal.
    - `debug`: If `True`, will interactively plot debug information during alignment
               (independently, plots are saved to disk).
    - `prefix`: Optional prefix for debug plot filenames.
    """
    ref_sample_rate: int
    target_sample_rate: int
    debug: bool = True
    prefix: Optional[str] = None

    def get_time_points(self, ref_signal, target_signal):
        ref_times = get_time_points(ref_signal)
        target_times = get_time_points(target_signal)

        return ref_times, target_times

    def crop_signal(self, ref_signal, target_signal, align_params: AlignmentParams):
        lag_time, target_sample_rate = align_params.lag, align_params.sample_rate
        lag_time = maybe(lag_time, 0)
        target_sample_rate = maybe(target_sample_rate, self.target_sample_rate)
        if lag_time > 0: # target starts first
            target_signal = target_signal[int(lag_time * target_sample_rate):]
        elif lag_time < 0: # reference starts first
            ref_signal = ref_signal[int(-lag_time * self.ref_sample_rate):]

        return ref_signal, target_signal

    def align(self, ref_signal, target_signal, align_params = (None, None, None)):
        raise NotImplementedError("Attempted to directly use BaseAligner. "
                                  "Instead, inherit from this class to implement "
                                  "a specific alignment protocol.")

@dataclass
class CrossCorrelationAligner(BaseAligner):
    """Aligns signals using cross-correlation
    (padding end of shorter signal with zeros).

    This is best used to determine a lag time between signals.
    The computed result is only valid when the peak cross-correlation
    is at least the same height as the minimum number of events in either signal.
    """
    def align(self, ref_signal, target_signal, align_params = AlignmentParams()):
        ref_signal, target_signal = self.crop_signal(ref_signal, target_signal,
                                                     align_params=align_params)
        ref_times, target_times = self.get_time_points(ref_signal, target_signal)
        nspikes = min(len(ref_times), len(target_times))
        target_signal = resample_signal(target_signal,
                                        source_rate=self.target_sample_rate,
                                        target_rate=self.ref_sample_rate)
        max_length = max(len(target_signal), len(ref_signal))
        target_signal = np.pad(target_signal, (0, max_length - len(target_signal)))
        ref_signal = np.pad(ref_signal, (0, max_length - len(ref_signal)))

        cross_corr = np.correlate(ref_signal, target_signal, mode="full")
        max_corr = np.max(cross_corr)
        peaks = np.argwhere(cross_corr == max_corr).flatten()
        mid_peak = np.argmin(np.abs(peaks - max_length))
        lag_idx = peaks[mid_peak] - max_length + 1
        lag_time = float(lag_idx / self.target_sample_rate)
        print("cross correlation lag time: ", lag_time)

        if self.debug:
            fig, ax = plt.subplots()
            ax.plot(np.arange(-max_length + 1, max_length), cross_corr)
            ax.axvline(lag_idx, color="r", linestyle="--", label=f"Lag: {lag_idx}")
            ax.set_xlabel("Lag Time (frames)")
            ax.set_ylabel("Cross-Correlation")
            ax.set_title('Cross-correlation between LED times and analog sync times')
            ax.legend()
        else:
            fig = None

        align_params = replace(align_params,
                               lag=lag_time, good=(max_corr == nspikes))

        return align_params, fig

@dataclass
class RegressionAligner(BaseAligner):
    """Aligns signals by regressing event times from each signal onto each other.

    This assumes identity mapping between time points
    (after adjusting for existing alignment parameters).
    Both the lag time and sample rate are adjusted based on the regression.
    The alignment is valid if the RMSE is below a set threshold.

    Additional attributes:
    - `max_rmse`: Maximum root mean squared error for regression."""
    max_rmse: float = 1e-2

    def align(self, ref_signal, target_signal, align_params = AlignmentParams()):
        ref_signal, target_signal = self.crop_signal(ref_signal, target_signal,
                                                     align_params=align_params)
        ref_times, target_times = self.get_time_points(ref_signal, target_signal)
        target_sample_rate = maybe(align_params.sample_rate, self.target_sample_rate)

        min_length = min(len(target_times), len(ref_times))
        rtimes = ref_times[:min_length] / self.ref_sample_rate
        # time_diffs = times - target_times[:min_length] / self.target_sample_rate
        ttimes = target_times[:min_length] / target_sample_rate

        if len(rtimes) > 0 and len(ttimes) > 0:
            coefficients = np.polyfit(rtimes, ttimes, deg=1)
            ttimes_fitted = np.polyval(coefficients, rtimes)
            lag_slope, lag_time = coefficients
            lag_time = maybe(align_params.lag, 0) - lag_time
            rmse = np.sqrt(np.mean((ttimes_fitted - ttimes) ** 2))
            print("regression lag time: ", lag_time)
            print("regression slope (should be near 1): ", lag_slope)
            print("regression rmse (should be near 0): ", rmse)
        else:
            print("regresssion failed: too few time points")
            lag_slope, lag_time, rmse = 1, 0, np.inf
            ttimes_fitted = None

        if self.debug:
            fig, ax = plt.subplots()
            ax.scatter(rtimes, ttimes)
            if ttimes_fitted is not None:
                ax.plot(rtimes, ttimes_fitted)
            ax.set_xlabel("Reference Signal On-Times (sec)")
            ax.set_ylabel("Target Signal On-Times (sec)")
        else:
            fig = None

        align_params = replace(align_params,
                               lag=(lag_slope * lag_time),
                               sample_rate=(lag_slope * target_sample_rate),
                               slope=lag_slope,
                               good=(rmse < self.max_rmse))

        return align_params, fig

@dataclass
class SampleRateAligner(BaseAligner):
    """Aligns signals by matching their sample rates.

    This estimates sample rates based on the inter-event intervals (in seconds).
    First, each signals inter-event interval is estimated using the median.
    Second, the matching ratio is computed as the ratio between the estimates.
    Finally, the sample rate of the target is adjusted by this ratio.

    The debug plot shows the histogram of the difference in matched intervals.
    When there are one or two modes in the distribution, the alignment works well.
    """
    def align(self, ref_signal, target_signal, align_params = AlignmentParams()):
        ref_signal, target_signal = self.crop_signal(ref_signal, target_signal,
                                                     align_params=align_params)
        ref_times, target_times = self.get_time_points(ref_signal, target_signal)

        min_length = min(len(target_times), len(ref_times))
        target_times = target_times[:min_length]
        ref_times = ref_times[:min_length]
        target_sample_rate = maybe(align_params.sample_rate, self.target_sample_rate)
        target_gaps = np.diff(target_times) / target_sample_rate
        ref_gaps = np.diff(ref_times) / self.ref_sample_rate
        gap_ratio = np.median(target_gaps) / np.median(ref_gaps)
        true_sample_rate = gap_ratio * target_sample_rate
        print("true sample rate: ", true_sample_rate)

        if self.debug:
            fig, ax = plt.subplots()
            ax.hist(target_gaps - ref_gaps, bins=50)
            ax.set_xlabel("Difference of inter-pulse intervals (sec)")
            ax.set_ylabel("Count")
            fig.tight_layout()
        else:
            fig = None

        align_params = replace(align_params,
                               lag=(maybe(align_params.lag, 0) * gap_ratio),
                               sample_rate=true_sample_rate,
                               good=(not np.isnan(true_sample_rate)))

        return align_params, fig
