import asyncio

import numpy as np

from src.ppmbin import BinPPMSymbol

class RealPPMSymbol(np.ndarray):
    def __new__(cls, size:int, index:int, samples_per_bit: int, power:float):
        """Create a new RealPPMSymbol"""
        obj = np.asarray(np.zeros(size, dtype=float))
        obj[index] = power
        return np.repeat(obj, samples_per_bit).view(cls)
    
    @classmethod
    def fromBin(cls, bin_ppm:BinPPMSymbol, samples_per_bit: int, power:float):
        expanded = np.repeat(bin_ppm.astype(float), samples_per_bit)
        return (expanded * power).view(cls)
    
class RealPPMGen:
    def __init__(self, samples_per_bit: int, power: float):
        self.samples_per_bit = samples_per_bit
        self.power = power

    def from_binchunk(self, binchunk: np.ndarray):
        """Convert array of BinPPMSymbols to array of RealPPMSymbols."""
        result = np.empty(len(binchunk), dtype=object)
        for i, bin_sym in enumerate(binchunk):
            result[i] = RealPPMSymbol.fromBin(bin_sym, self.samples_per_bit, self.power)
        return result

    async def process(self, in_queue: asyncio.Queue, out_queue: asyncio.Queue):
        """Consume BinPPMSymbol arrays, produce RealPPMSymbol equivalents."""
        while True:
            chunk = await in_queue.get()
            if chunk is None:
                await out_queue.put(None)
                break
            real_chunk = self.from_binchunk(chunk)
            await out_queue.put(real_chunk)


def make_tx_pulse(shape: str, sps: int) -> np.ndarray:
    return np.ones((sps))


def make_pulse_template(shape: str, sps: int, neg_len: int | None = None) -> np.ndarray:
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
