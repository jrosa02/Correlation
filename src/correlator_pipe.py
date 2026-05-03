import functools
from typing import Any, Literal

import numpy as np
from numpy import dtype, ndarray

from src.signal_pipe import SignalPipe

class CorrPipe(SignalPipe):
    def __init__(self, ref_type: Literal['rect', 'triangle'] = 'rect', pulse_width = 4, seed: int = 42) -> None:
        super().__init__(seed)
        self.ref_type = ref_type
        self.pulse_width = pulse_width

    @functools.lru_cache(maxsize=32)
    def get_reference(self, reference: Literal['rect'] = 'rect', pulse_width: int = 4) -> np.ndarray:
        match self.ref_type:
            case 'rect':
                return np.concat((np.repeat([-1], pulse_width//2-1), [-2, 2], np.repeat([1], pulse_width-2), [2, -2], np.repeat([-1], pulse_width//2-1)), dtype=np.float64)
            case 'triangle':
                arr = np.concatenate(([i for i in range(pulse_width)], [pulse_width-i for i in range(pulse_width)]), dtype=np.float64)
                return (arr - arr.mean()) / arr.std()
            case _:
                raise ValueError()

    def process(self, signal: ndarray[tuple[Any, ...], dtype[Any]]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        ref = self.get_reference(self.ref_type, self.pulse_width)
        rows = signal.reshape(-1, signal.shape[-1]) if signal.ndim > 1 else signal[np.newaxis]
        result = np.divide(np.stack([np.correlate(row.astype(np.float64), ref, 'same') for row in rows]), len(ref))
        return result if signal.ndim > 1 else result[0]
    
    def reset(self) -> None:
        pass
