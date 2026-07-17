from __future__ import annotations

import numpy as np


def moving_average(values, window_bins: int | None) -> np.ndarray:
    """Centered convolution average shared by EMG and firing-rate plots."""
    array = np.asarray(values)
    if window_bins is None or window_bins <= 1:
        return array
    size = int(window_bins)
    kernel = np.ones(size, dtype=float) / float(size)
    return np.convolve(array.astype(float), kernel, mode="same")
