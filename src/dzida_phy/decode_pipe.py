from typing import Any

import numpy as np
from numpy import dtype, ndarray

from dzida_phy.physical_units import Quantity
from dzida_phy.signal_pipe import SignalPipe, CompoundPipe, Terminator
from dzida_phy.plot_pipe import PlotPipe, PlotInput


class DecodeSink_Simple(SignalPipe):
    DOWNSAMPLE_BIAS: float = 0.8

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
        return np.floor(offsets / self.sampling_rate).astype(np.int64)

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


class DecodeSink_Timed(DecodeSink_Simple):
    def __init__(
        self,
        data_len: int,
        chunk_size: int,
        n_slots: int,
        sample_rate: Quantity,
        slot_rate: Quantity,
        seed: int = 42,
    ) -> None:
        super().__init__(
            data_len=data_len,
            chunk_size=chunk_size,
            n_slots=n_slots,
            sampling_rate=sample_rate.to_hz() / slot_rate.to_hz(),
            seed=seed,
        )


class DecodePipe_Timed(SignalPipe):
    """Decode pipe (process-style): decodes signal, stores indices, returns one-hot symbols."""

    def __init__(self, data_len: int, chunk_size: int, n_slots: int,
                 sample_rate: Quantity, slot_rate: Quantity, seed: int = 42) -> None:
        super().__init__(seed)
        self.n_slots = n_slots
        self.chunk_size = chunk_size
        self.sampling_rate = sample_rate.to_hz() / slot_rate.to_hz()
        self._index = 0
        self.data = np.full(data_len, -1, dtype=np.int64)

    def process(self, signal: np.ndarray) -> np.ndarray:
        rows = signal.reshape(-1, signal.shape[-1]) if signal.ndim > 1 else signal[np.newaxis]
        offsets = np.argmax(rows, axis=1)
        indices = np.floor(offsets / self.sampling_rate).astype(np.int64)
        self.data[self._index:self._index + self.chunk_size] = indices
        self._index += self.chunk_size
        symbols = np.zeros((len(indices), self.n_slots), dtype=np.uint8)
        for i, idx in enumerate(indices):
            if 0 <= idx < self.n_slots:
                symbols[i, idx] = 1
        return symbols

    @property
    def decoded_data(self) -> np.ndarray:
        return self.data.copy()

    def reset(self) -> None:
        self._index = 0
        self.data[:] = -1


class DecodePlotSink_Timed(CompoundPipe):
    """Decode + bar-plot compound sink. get_data returns decoded symbol indices."""

    def __init__(self, data_len: int, chunk_size: int, n_slots: int,
                 sample_rate: Quantity, slot_rate: Quantity,
                 plot_input: PlotInput | None = None, seed: int = 42) -> None:
        self._decoder = DecodePipe_Timed(data_len, chunk_size, n_slots,
                                         sample_rate, slot_rate, seed)
        plotpipe = PlotPipe(plot_input, 'bar', title='Decoded PPM | FPGA',
                            sample_rate=None, seed=seed) if plot_input else None
        super().__init__([self._decoder, plotpipe, Terminator()], seed)

    @property
    def decoded_data(self) -> np.ndarray:
        return self._decoder.decoded_data
