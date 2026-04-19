import numpy as np


def generate_bits(n_bits: int, min_gap: int = 0, seed: int | None = None) -> np.ndarray:
    """Generate sparse binary sequence with guaranteed min_gap zeros between any two ones."""
    rng = np.random.default_rng(seed)
    bits = np.zeros(n_bits, dtype=np.float64)
    # Place ones greedily with vectorized rejection: draw candidates, enforce spacing
    i = 0
    while i < n_bits:
        if rng.random() < 0.5:
            bits[i] = 1.0
            i += min_gap + 1  # skip min_gap slots after a one
        else:
            i += 1
    return bits


def make_tx_pulse(shape: str, sps: int, **kwargs) -> np.ndarray:
    """Unipolar TX pulse: sps ones at centre, energy-normalised to 1. Length = sps."""
    pulse = np.zeros(sps)
    pulse[:] = 0.0
    if shape == "rect":
        pulse[:] = 1.0
    elif shape == "gaussian":
        t = np.arange(sps) - (sps - 1) / 2
        sigma = sps / 4
        pulse[:] = np.exp(-0.5 * (t / sigma) ** 2)
    elif shape == "raised_cosine":
        t = np.arange(sps)
        pulse[:] = 0.5 * (1 - np.cos(2 * np.pi * t / sps))
    elif shape == "delta":
        pulse[sps // 2] = 1.0
    else:
        raise ValueError(f"Unknown pulse shape: {shape!r}")
    return pulse


def make_pulse_template(shape: str, sps: int, neg_len: int | None = None, **kwargs) -> np.ndarray:
    """Correlation template: [neg_val]*neg_len + [1.0]*sps + [neg_val]*neg_len.

    neg_len: negative wing samples per side (even total length = sps + 2*neg_len).
             Defaults to sps.
    neg_val: snapped to nearest power of 2 so that sum == 0:
             neg_val = sps / (2 * neg_len).
    All non-rect shapes use the same structure with a raised-cosine positive lobe.
    """
    if neg_len is None:
        neg_len = sps
    neg_len = max(1, int(neg_len))
    # zero-sum: sps * 1 + 2 * neg_len * neg_val = 0 → neg_val = -sps / (2*neg_len)
    exact = sps / (2 * neg_len)
    # snap to nearest power of 2
    neg_val = -(2 ** round(np.log2(exact)))

    if shape in ("rect", "delta"):
        pos = np.ones(sps)
    elif shape == "gaussian":
        t = np.arange(sps) - (sps - 1) / 2
        sigma = sps / 4
        pos = np.exp(-0.5 * (t / sigma) ** 2)
        pos /= pos.max()
    elif shape == "raised_cosine":
        t = np.arange(sps)
        pos = 0.5 * (1 - np.cos(2 * np.pi * t / sps))
    else:
        raise ValueError(f"Unknown pulse shape: {shape!r}")

    pulse = np.concatenate([
        np.full(neg_len, neg_val),
        pos,
        np.full(neg_len, neg_val),
    ])
    pulse -= pulse.mean()  # enforce exact zero sum
    return pulse


def build_signal(bits: np.ndarray, template: np.ndarray, sps: int) -> np.ndarray:
    """Build transmit waveform via vectorized convolution.

    Expands bits to an impulse train at sample rate, then convolves with template.
    """
    impulse_train = np.zeros(len(bits) * sps)
    impulse_train[::sps] = bits  # place bit values at symbol start positions
    return np.convolve(impulse_train, template, mode="full")[: len(bits) * sps]
