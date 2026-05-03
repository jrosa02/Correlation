from typing import Any

import numpy as np
from numpy import dtype, ndarray
from scipy import signal as sp

from src.signal_pipe import SignalPipe


class BandpassPipe(SignalPipe):

    def __init__(self, low:float = 1e-3, high:float = 1.0, order:int = 4, seed: int = 42) -> None:
        super().__init__(seed)
        self.low = low
        self.high = high
        self.order = order

    def bandpass_filter(self, sig: np.ndarray) -> np.ndarray:
        sos = sp.butter(self.order, [self.low, self.high], btype="bandpass", output="sos")
        return sp.sosfiltfilt(sos, sig)

    def process(self, signal: ndarray[tuple[Any, ...], dtype[Any]]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        return self.bandpass_filter(signal)
    
    def reset(self) -> None:
        pass
