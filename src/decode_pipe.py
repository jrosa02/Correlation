from typing import Any

import numpy as np
from numpy import dtype, ndarray

from src.signal_pipe import SignalPipe


class DecodeSink(SignalPipe):
    def __init__(self, data_len: int, chunk_size: int, n_slots: int, sampling_rate: float, seed: int = 42) -> None:
        super().__init__(seed)
        self.n_slots = n_slots
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self._index = 0
        self.data = np.full(data_len, -1, dtype=np.int64)

    @staticmethod
    def retrieve_offsets(signal: np.ndarray) -> np.ndarray:
        rows = signal.reshape(-1, signal.shape[-1]) if signal.ndim > 1 else signal[np.newaxis]
        return np.argmax(rows, axis=1)

    def downsample(self, offsets: np.ndarray) -> np.ndarray:
        return np.round(offsets / self.sampling_rate -0.3).astype(np.int64)

    def consume(self, signal: ndarray[tuple[Any, ...], dtype[Any]]):
        offsets = self.retrieve_offsets(signal)
        data = self.downsample(offsets)
        self.data[self._index:self._index + self.chunk_size] = data
        self._index += self.chunk_size

    @property
    def get_data(self):
        return self.data.copy()
    
    def reset(self) -> None:
        self._index = 0    
