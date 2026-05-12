from typing import Any, Literal

import numpy as np
from numpy import dtype, ndarray

from src.native_optimized import correlate
from src.signal_pipe import SignalPipe

_REF_TYPE = {'rect': 0, 'triangle': 1}


class CorrPipe(SignalPipe):
    def __init__(self, ref_type: Literal['rect', 'triangle'] = 'rect', pulse_width=4, seed: int = 42) -> None:
        super().__init__(seed)
        self.ref_type = ref_type
        self.pulse_width = pulse_width

    def process(self, signal: ndarray[tuple[Any, ...], dtype[Any]]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        rows = signal.reshape(-1, signal.shape[-1]) if signal.ndim > 1 else signal[np.newaxis]
        rows = np.ascontiguousarray(rows, dtype=np.float64)
        out = np.empty_like(rows)
        correlate(rows, _REF_TYPE[self.ref_type], self.pulse_width, out)
        return out if signal.ndim > 1 else out[0]

    def reset(self) -> None:
        pass
