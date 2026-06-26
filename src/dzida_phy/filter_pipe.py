from typing import Any

import numpy as np
from numpy import dtype, ndarray
from scipy import signal as sp

from dzida_phy.physical_units import Quantity
from dzida_phy.signal_pipe import SignalPipe, CompoundPipe
from dzida_phy.plot_pipe import PlotPipe, PlotInput


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


class LowpassPipe_Simple(SignalPipe):

    def __init__(self, cutoff: float = 0.5, order: int = 4, seed: int = 42) -> None:
        super().__init__(seed)
        self.cutoff = cutoff
        self.order = order

    def lowpass_filter(self, sig: np.ndarray) -> np.ndarray:
        # Clamp cutoff to valid range (0, 1) for safety
        cutoff = max(0.0001, min(0.9999, float(self.cutoff)))
        sos = sp.butter(self.order, cutoff, btype="low", output="sos")
        filtered = sp.sosfilt(sos, sig)
        assert isinstance(filtered, np.ndarray)
        return filtered

    def process(self, signal: ndarray[tuple[Any, ...], dtype[Any]]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        return self.lowpass_filter(signal)

    def reset(self) -> None:
        pass


class LowpassPipe_Timed(LowpassPipe_Simple):

    def __init__(
        self,
        lowpass_cutoff: Quantity,
        sample_rate: Quantity,
        order: int = 4,
        seed: int = 42,
    ) -> None:
        nyquist = sample_rate.to_hz() / 2.0
        super().__init__(
            cutoff=lowpass_cutoff.to_hz() / nyquist,
            order=order,
            seed=seed,
        )


class HighpassPipe_Simple(SignalPipe):

    def __init__(self, cutoff: float = 0.1, order: int = 4, seed: int = 42) -> None:
        super().__init__(seed)
        self.cutoff = cutoff
        self.order = order

    def highpass_filter(self, sig: np.ndarray) -> np.ndarray:
        sos = sp.butter(self.order, self.cutoff, btype="high", output="sos")
        filtered = sp.sosfilt(sos, sig)
        assert isinstance(filtered, np.ndarray)
        return filtered

    def process(self, signal: ndarray[tuple[Any, ...], dtype[Any]]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        return self.highpass_filter(signal)

    def reset(self) -> None:
        pass


class HighpassPipe_Timed(HighpassPipe_Simple):

    def __init__(
        self,
        highpass_cutoff: Quantity,
        sample_rate: Quantity,
        order: int = 4,
        seed: int = 42,
    ) -> None:
        nyquist = sample_rate.to_hz() / 2.0
        super().__init__(
            cutoff=highpass_cutoff.to_hz() / nyquist,
            order=order,
            seed=seed,
        )


class HighpassModule_Timed(CompoundPipe):
    def __init__(self, highpass_cutoff: Quantity, sample_rate: Quantity, plot_input: PlotInput | None = None,
                 order: int = 4, seed: int = 42) -> None:
        filt     = HighpassPipe_Timed(highpass_cutoff, sample_rate, order, seed)
        highpass_cutoff.set_repr('freq')
        plotpipe = PlotPipe(plot_input, title=f"Highpass - Cufoff: {highpass_cutoff}", sample_rate=sample_rate) if plot_input else None
        super().__init__([filt, plotpipe], seed)


class LowpassModule_Timed(CompoundPipe):
    def __init__(self, lowpass_cutoff: Quantity, sample_rate: Quantity,
                 plot_input: PlotInput | None = None,
                 order: int = 4, seed: int = 42) -> None:
        filt     = LowpassPipe_Timed(lowpass_cutoff, sample_rate, order, seed)
        lowpass_cutoff.set_repr('freq')
        plotpipe = PlotPipe(plot_input, title=f"Lowpass - Cutoff: {lowpass_cutoff}", sample_rate=sample_rate) if plot_input else None
        super().__init__([filt, plotpipe], seed)


class BandpassModule_Timed(CompoundPipe):
    def __init__(self, bandpass_low: Quantity, bandpass_high: Quantity,
                 sample_rate: Quantity,
                 plot_input: PlotInput | None = None,
                 order: int = 4, seed: int = 42) -> None:
        filt     = BandpassPipe_Timed(bandpass_low, bandpass_high, sample_rate, order, seed)
        bandpass_low.set_repr('freq')
        bandpass_high.set_repr('freq')
        plotpipe = PlotPipe(plot_input, title=f"Bandpass - {bandpass_low} / {bandpass_high}", sample_rate=sample_rate) if plot_input else None
        super().__init__([filt, plotpipe], seed)
