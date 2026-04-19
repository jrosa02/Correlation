import numpy as np


def add_awgn(signal: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    """Add AWGN at the given SNR (dB). sigma^2 = signal_power / SNR_linear."""
    snr = 10 ** (snr_db / 10)
    sigma2 = 0.5 / snr  # reference: pulse amplitude=1, 50% bit probability
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, np.sqrt(sigma2), size=signal.shape)
    return signal + noise
