import functools
from typing import Any, Literal

import numpy as np
from numpy import dtype, ndarray

from src.signal_pipe import SignalPipe

class MaximumPipe(SignalPipe):
    def __init__(self, rate: int, seed: int = 42) -> None:
        super().__init__(seed)
        self.rate = rate
        self.piramid = np.concatenate(([i for i in range(rate//2)], [rate//2-i for i in range(rate//2)]), dtype=int)

    def get_max_val_offset(self, signal: np.ndarray) -> np.ndarray:
        return np.argmax(signal, axis=1)

    def process(self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> np.ndarray:
        offsets = self.get_max_val_offset(signal)
        n_rows, n_cols = signal.shape
        out = np.zeros((n_rows, n_cols), dtype=np.float64)
        out[np.arange(n_rows), offsets] = 1.0
        return out

