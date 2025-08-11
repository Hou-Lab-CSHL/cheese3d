import numpy as np

def resample_signal(signal, source_rate, target_rate):
    if source_rate > target_rate:
        return signal[::int(source_rate / target_rate)]
    else:
        return np.repeat(signal, int(target_rate / source_rate))

def get_time_points(signal):
    return np.where(np.diff(signal) > 0)[0]
