from typing import Any, Literal

import numpy as np
from numba import njit, prange

from dzida_phy.physical_units import Quantity
from dzida_phy.signal_pipe import SignalPipe


@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def _repeat_upsample(signal, rate):
    n_rows, n_cols = signal.shape
    out = np.empty((n_rows, n_cols * rate), signal.dtype)
    for r in prange(n_rows):
        out[r] = np.repeat(signal[r], rate)
    return out


class UpSampler_Simple(SignalPipe):
    def __init__(self, rate: int, method: Literal['repeat'] = 'repeat', seed: int = 42) -> None:
        super().__init__(seed)
        self.rate = rate
        self.method = method

    def process(self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        match self.method:
            case 'repeat':
                return _repeat_upsample(signal, self.rate)
            case _:
                raise NotImplementedError()

    def reset(self) -> None:
        pass


class UpSampler_Timed(UpSampler_Simple):
    def __init__(
        self,
        sample_rate: Quantity,
        slot_rate: Quantity,
        method: Literal['repeat'] = 'repeat',
        seed: int = 42,
    ) -> None:
        super().__init__(
            rate=round(sample_rate.to_hz() / slot_rate.to_hz()),
            method=method,
            seed=seed,
        )
