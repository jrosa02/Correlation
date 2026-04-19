import numpy as np


def add_awgn(signal: np.ndarray, eb_n0_db: float, sps: int) -> np.ndarray:
    """Add AWGN scaled to the given Eb/N0 (dB).

    Noise variance: sigma^2 = E_b / (2 * sps * Eb_N0_linear)
    where E_b = mean symbol energy per bit = signal power * sps.
    """
    eb_n0 = 10 ** (eb_n0_db / 10)
    signal_power = np.mean(signal ** 2)
    # N0/2 = sigma^2; E_b = signal_power * sps (energy per bit at sps samples/bit)
    sigma2 = signal_power * sps / (2 * eb_n0) if signal_power > 0 else 1e-12
    noise = np.random.default_rng().normal(0, np.sqrt(sigma2), size=signal.shape)
    return signal + noise
