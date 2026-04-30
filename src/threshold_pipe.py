import functools
from typing import Any, Literal

import numpy as np
from numpy import dtype, ndarray

from src.signal_pipe import SignalPipe

class ThresholdPipe(SignalPipe):
    def __init__(self, threshold:float = 0.5, seed: int = 42) -> None:
        super().__init__(seed)
        self.threshold = threshold

    def process(self, signal: ndarray[tuple[Any, ...], dtype[Any]]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        return signal>self.threshold

