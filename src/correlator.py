import numpy as np


def sliding_correlator(
    signal: np.ndarray,
    template: np.ndarray,
    window_len: int,
    sps: int,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cross-correlate signal with a windowed pulse template.

    threshold: decision level. If None, auto-set to midpoint of sample range.
               Pass an explicit value to enable external optimisation of this parameter.

    Returns (corr_output, decisions, threshold_used).
    """
    ref = template[:window_len]
    corr = np.correlate(signal, ref, mode="full")
    delay = len(ref) - 1
    corr_aligned = np.zeros(len(signal))
    valid = corr[delay : delay + len(signal)]
    corr_aligned[:len(valid)] = valid
    corr_aligned[:delay] = 0.0  # zero samples where template hasn't fully entered

    sample_idx = np.arange(0, len(corr_aligned), sps)
    samples = corr_aligned[sample_idx]

    threshold = float(0.5 * (samples.max() + samples.min())) if threshold is None else float(threshold)
    decisions = (samples > threshold).astype(np.float64)
    return corr_aligned, decisions, threshold
