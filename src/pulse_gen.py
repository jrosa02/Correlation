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


def make_tx_pulse(shape: str, sps: int, duty_cycle: float = 0.5, **kwargs) -> np.ndarray:
    """Return a unipolar (≥ 0), symmetric TX pulse of length sps.

    Represents a physical laser pulse: zero outside the active region,
    positive within — so bits 0,1,0 upsample to 0…0, +…+, 0…0.
    """
    sps = sps if sps % 2 == 0 else sps + 1
    half_width = max(1, int(duty_cycle * sps / 2))
    center = sps // 2
    pulse = np.zeros(sps)
    t_local = np.arange(2 * half_width) - half_width + 0.5  # symmetric around 0

    if shape == "rect":
        pulse[center - half_width: center + half_width] = 1.0

    elif shape == "gaussian":
        sigma = half_width / 2
        pulse[center - half_width: center + half_width] = np.exp(-0.5 * (t_local / sigma) ** 2)

    elif shape == "raised_cosine":
        pulse[center - half_width: center + half_width] = (
            0.5 * (1 - np.cos(np.pi * (t_local + half_width) / half_width))
        )

    elif shape == "delta":
        pulse[center] = 1.0

    else:
        raise ValueError(f"Unknown pulse shape: {shape!r}")

    energy = np.sum(pulse ** 2)
    return pulse / np.sqrt(energy) if energy > 0 else pulse


def make_pulse_template(shape: str, sps: int, duty_cycle: float = 0.5, **kwargs) -> np.ndarray:
    """Return a zero-mean, symmetric pulse template of length sps.

    All shapes are symmetric around the centre sample and have integral ≈ 0,
    resembling  -1 -1 | +1 +1 +1 +1 | -1 -1  (bipolar / wavelet-like).

    shape: "rect" | "gaussian" | "raised_cosine" | "delta"
    duty_cycle: fraction of sps occupied by the positive centre lobe
    """
    sps = sps if sps % 2 == 0 else sps + 1  # ensure even length for symmetry
    center = sps / 2
    t = np.arange(sps) - center + 0.5   # symmetric around 0: [-(sps/2)+.5 … (sps/2)-.5]
    half_width = max(1, duty_cycle * sps / 2)

    if shape == "rect":
        # Positive centre lobe, negative wings — like -1…-1 +1…+1 -1…-1
        wing = int((sps - 2 * int(half_width)) // 2)
        pulse = np.full(sps, -1.0)
        pulse[wing: sps - wing] = 1.0

    elif shape == "gaussian":
        # Mexican-hat (Ricker wavelet): second derivative of Gaussian
        sigma = half_width / np.sqrt(2)
        z = (t / sigma) ** 2
        pulse = (1 - z) * np.exp(-z / 2)

    elif shape == "raised_cosine":
        # Full cosine cycle: naturally zero-mean and symmetric
        pulse = np.cos(2 * np.pi * t / sps)

    elif shape == "delta":
        # Bipolar doublet centred: −1 at −0.5, +1 at +0.5 (one sample each side)
        pulse = np.zeros(sps)
        pulse[sps // 2 - 1] = -1.0
        pulse[sps // 2]     =  1.0

    else:
        raise ValueError(f"Unknown pulse shape: {shape!r}")

    # Enforce exact zero mean (removes any floating-point residual)
    pulse -= pulse.mean()

    energy = np.sum(pulse ** 2)
    return pulse / np.sqrt(energy) if energy > 0 else pulse


def build_signal(bits: np.ndarray, template: np.ndarray, sps: int) -> np.ndarray:
    """Build transmit waveform via vectorized convolution.

    Expands bits to an impulse train at sample rate, then convolves with template.
    """
    impulse_train = np.zeros(len(bits) * sps)
    impulse_train[::sps] = bits  # place bit values at symbol start positions
    return np.convolve(impulse_train, template, mode="full")[: len(bits) * sps]
