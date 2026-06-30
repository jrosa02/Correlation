from typing import Any

import numpy as np
from numba import njit, prange

from dzida_phy.signal_pipe import SignalPipe

_ROW_BATCH_NDIM = 2


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _add_noise(signal, std, rng):
    return signal + rng.normal(0.0, std, signal.shape)


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _add_brownian_noise(signal, std, rng):
    white = rng.normal(0.0, std, signal.shape)
    out = np.empty_like(signal)
    for i in prange(signal.shape[0]):
        brown = np.cumsum(white[i])
        out[i] = signal[i] + brown
    return out


class AWGN(SignalPipe):
    def __init__(self, linear_noise_power, seed: int = 42) -> None:
        super().__init__(seed)
        self.noise_power = linear_noise_power
        self.std = np.sqrt(linear_noise_power)

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        return _add_noise(signal, self.std, self.rng)

    def process(
        self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        return self.add_noise(signal)

    def reset(self) -> None:
        pass


class BrownianNoise(SignalPipe):
    def __init__(self, linear_noise_power: float | None, seed: int = 42) -> None:
        super().__init__(seed)
        self.noise_power = linear_noise_power if linear_noise_power is not None else 0
        self.std = np.sqrt(linear_noise_power) if linear_noise_power is not None else 0

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        s = signal if signal.ndim == _ROW_BATCH_NDIM else signal[np.newaxis]
        out = _add_brownian_noise(s, self.std, self.rng)
        return out if signal.ndim == _ROW_BATCH_NDIM else out[0]

    def process(
        self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        return self.add_noise(signal)

    def reset(self) -> None:
        pass
