import numpy as np


def sliding_correlator(
    signal: np.ndarray,
    template: np.ndarray,
    window_len: int,
    sps: int,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Cross-correlate signal with a windowed pulse template.

    threshold: decision level. If None, auto-set to midpoint of sample range.
               Pass an explicit value to enable external optimisation of this parameter.

    Returns (corr_output, decisions, threshold_used, peak_indices).
    peak_indices[i] is the sample index of the correlation argmax within symbol window i.
    """
    ref = template[:window_len]
    # pad by the index of the positive peak so that peak for bit i lands at i*sps
    pad = int(np.argmax(ref))
    padded = np.concatenate([np.zeros(pad), signal])
    corr = np.correlate(padded, ref, mode="full")
    corr_aligned = corr[len(ref) - 1 : len(ref) - 1 + len(signal)]

    n_sym = len(corr_aligned) // sps
    windows = corr_aligned[:n_sym * sps].reshape(n_sym, sps)
    peak_offsets = np.argmax(windows, axis=1)
    peak_indices = np.arange(n_sym) * sps + peak_offsets
    window_maxes = windows.max(axis=1)

    # threshold from global corr max — same value shown as line in the plot
    threshold = float(0.6 * corr_aligned.max()) if threshold is None else float(threshold)
    decisions = (window_maxes > threshold).astype(np.float64)

    return corr_aligned, decisions, threshold, peak_indices
