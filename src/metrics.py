import numpy as np


def compute_ber(bits_tx: np.ndarray, bits_rx: np.ndarray, sps: int | None = None, peak_indices: np.ndarray | None = None) -> float:
    """Compute BER via offset comparison when peak_indices provided, else bit-by-bit.

    Offset mode: compare detected pulse positions (rounded) vs true pulse positions.
    Errors = positions in symmetric difference of the two sets.
    """
    if sps is not None and peak_indices is not None:
        # symbol-domain: round detected peak positions to nearest symbol index
        true_syms = set(np.where(bits_tx)[0])
        detected_syms = set(np.round(peak_indices[bits_rx.astype(bool)] / sps).astype(int))
        errors = len(true_syms.symmetric_difference(detected_syms))
        total = max(len(true_syms), 1)
        return float(errors / total)
    else:
        # bit-by-bit fallback
        n = min(len(bits_tx), len(bits_rx))
        return float(np.mean(bits_tx[:n] != bits_rx[:n]))


def compute_output_snr(clean: np.ndarray, received: np.ndarray) -> float:
    noise = received - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)
