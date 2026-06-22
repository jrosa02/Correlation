from typing import Any

import numpy as np
from numpy import dtype, ndarray
from numba import vectorize

from dzida_phy.signal_pipe import SignalPipe


@vectorize(['b1(f8, f8)'], target='parallel', cache=True)
def _threshold(x, thresh):
    return x > thresh


class ThresholdPipe(SignalPipe):
    def __init__(self, threshold: float = 0.5, seed: int = 42) -> None:
        super().__init__(seed)
        self.threshold: float = threshold

    def process(self, signal: ndarray[tuple[Any, ...], dtype[Any]]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        return _threshold(signal.astype(np.float64), self.threshold)

    def reset(self) -> None:
        pass

