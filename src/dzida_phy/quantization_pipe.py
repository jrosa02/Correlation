import numpy as np

from dzida_phy.signal_pipe import SignalPipe


class QuantizatorPipe(SignalPipe):
    def __init__(self, bits: int, v_range: tuple[float, float] | None = None, seed: int = 42) -> None:
        super().__init__(seed)
        self.bits = bits
        self.levels = 2**bits
        self.v_range = v_range

    def process(self, signal: np.ndarray) -> np.ndarray:
        if self.v_range is not None:
            v_min, v_max = self.v_range
            signal = np.clip(signal, v_min, v_max)
        else:
            v_min, v_max = signal.min(), signal.max()
        step = (v_max - v_min) / self.levels
        if step == 0:
            return signal
        return np.floor((signal - v_min) / step) * step + v_min + step / 2

    def reset(self) -> None:
        pass
