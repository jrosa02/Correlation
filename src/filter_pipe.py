from typing import Any

import numpy as np
from numpy import dtype, ndarray
from scipy import signal as sp

from src.physical_units import Quantity
from src.signal_pipe import SignalPipe


class BandpassPipe_Simple(SignalPipe):

    def __init__(self, low: float = 1e-3, high: float = 1.0, order: int = 4, seed: int = 42) -> None:
        super().__init__(seed)
        self.low = low
        self.high = high
        self.order = order

    def bandpass_filter(self, sig: np.ndarray) -> np.ndarray:
        sos = sp.butter(self.order, [self.low, self.high], btype="bandpass", output="sos")
        filtered = sp.sosfilt(sos, sig)
        assert isinstance(filtered, np.ndarray)
        return filtered

    def process(self, signal: ndarray[tuple[Any, ...], dtype[Any]]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        return self.bandpass_filter(signal)

    def reset(self) -> None:
        pass


class BandpassPipe_Timed(BandpassPipe_Simple):

    def __init__(
        self,
        bandpass_low: Quantity,
        bandpass_high: Quantity,
        sample_rate: Quantity,
        order: int = 4,
        seed: int = 42,
    ) -> None:
        nyquist = sample_rate.to_hz() / 2.0
        super().__init__(
            low=bandpass_low.to_hz() / nyquist,
            high=bandpass_high.to_hz() / nyquist,
            order=order,
            seed=seed,
        )
