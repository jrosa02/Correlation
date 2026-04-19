import numpy as np
from scipy import signal as sp


def bandpass_filter(
    sig: np.ndarray,
    fs: float,
    f_center: float,
    bw: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter.

    f_center, bw: normalized frequencies (0–1, relative to Nyquist = fs/2).
    bw is the sweep axis in experiments.
    """
    nyq = fs / 2
    low = max(1e-4, (f_center - bw / 2))
    high = min(1 - 1e-4, (f_center + bw / 2))
    sos = sp.butter(order, [low, high], btype="bandpass", output="sos")
    return sp.sosfiltfilt(sos, sig)
