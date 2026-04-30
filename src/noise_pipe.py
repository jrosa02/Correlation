from typing import Any

import numpy as np
from numpy import dtype, ndarray

from src.signal_pipe import SignalPipe


class AWGN(SignalPipe):
    def __init__(self, noise_power, seed: int | None = None) -> None:
        super().__init__(seed)
        self.noise_power = noise_power
        self.noise = None

    def add_noise(self, signal: ndarray):
        self.noise = self.rng.normal(0, np.sqrt(self.noise_power), signal.shape)
        return signal + self.noise

    def process(self, signal: ndarray[tuple[Any, ...], dtype[Any]]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        return self.add_noise(signal)