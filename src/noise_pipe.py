from typing import Any

import numpy as np
from numba import njit

from src.signal_pipe import SignalPipe


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _add_noise(signal, std, rng):
    return signal + rng.normal(0.0, std, signal.shape)


class AWGN(SignalPipe):
    def __init__(self, noise_power, seed: int = 42) -> None:
        super().__init__(seed)
        self.noise_power = noise_power
        self.std = np.sqrt(noise_power)

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        return _add_noise(signal, self.std, self.rng)

    def process(self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        return self.add_noise(signal)

    def reset(self) -> None:
        pass
