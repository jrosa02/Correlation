from typing import Any, Literal

import numpy as np
from numpy import dtype, ndarray

from dzida_phy.native_optimized import correlate
from dzida_phy.physical_units import Quantity
from dzida_phy.signal_pipe import SignalPipe

_REF_TYPE = {'rect': 0, 'triangle': 1}


class CorrPipe_Simple(SignalPipe):
    def __init__(self, ref_type: Literal['rect', 'triangle'] = 'rect', pulse_width: int = 4, seed: int = 42) -> None:
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


class CorrPipe_Timed(CorrPipe_Simple):
    def __init__(
        self,
        sample_rate: Quantity,
        slot_rate: Quantity,
        ref_type: Literal['rect', 'triangle'] = 'rect',
        seed: int = 42,
    ) -> None:
        super().__init__(
            ref_type=ref_type,
            pulse_width=round(sample_rate.to_hz() / slot_rate.to_hz()),
            seed=seed,
        )
