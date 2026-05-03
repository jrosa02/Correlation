from typing import Any, Literal

import numpy as np

from src.signal_pipe import SignalPipe


class UpSampler(SignalPipe):
    def __init__(self, rate: int, method: Literal['repeat'] = 'repeat', seed: int = 42) -> None:
        super().__init__(seed)
        self.rate = rate
        self.method = method

    def upsample_repeat(self, signal: np.ndarray) -> np.ndarray:
        return np.repeat(signal, self.rate, axis=1)
    
    def process(self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        match self.method:
            case 'repeat':
                return self.upsample_repeat(signal)
            case _:
                raise NotImplementedError()
            
    def reset(self) -> None:
        pass