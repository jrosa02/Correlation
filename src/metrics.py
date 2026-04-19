import numpy as np


def compute_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    n = min(len(bits_tx), len(bits_rx))
    return float(np.mean(bits_tx[:n] != bits_rx[:n]))


def compute_output_snr(clean: np.ndarray, received: np.ndarray) -> float:
    noise = received - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)
