from typing import Any

import numpy as np
from numba import vectorize
from numpy import dtype, ndarray

from dzida_phy.fft_plot_pipe import FftPlotPipe
from dzida_phy.physical_units import Quantity
from dzida_phy.plot_pipe import PlotInput, PlotPipe
from dzida_phy.signal_pipe import CompoundPipe, SignalPipe


@vectorize(["b1(f8, f8)"], target="parallel", cache=True)
def _threshold(x, thresh):
    return x > thresh


class ThresholdPipe(SignalPipe):
    def __init__(self, threshold: float = 0.5, seed: int = 42) -> None:
        super().__init__(seed)
        self.threshold: float = threshold

    def process(self, signal: ndarray[tuple[Any, ...], dtype[Any]]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        return _threshold(signal.astype(np.float64), self.threshold)

    def reset(self) -> None:
        pass


class ThresholdModule(CompoundPipe):
    def __init__(
        self,
        threshold: float,
        sample_rate: Quantity,
        plot_input: PlotInput | None = None,
        fft_plot_input: PlotInput | None = None,
        seed: int = 42,
    ) -> None:
        self._threshold = threshold
        self._ax = plot_input.ax if plot_input else None
        pre_plot = (
            PlotPipe(
                plot_input,
                title="Threshold | FPGA",
                sample_rate=sample_rate,
                plot_kwargs={"alpha": 0.6, "color": "green"},
            )
            if plot_input
            else None
        )
        post_plot = (
            PlotPipe(plot_input, sample_rate=sample_rate, plot_kwargs={"alpha": 0.6, "color": "blue"})
            if plot_input
            else None
        )
        if pre_plot is not None:
            pre_plot.ax.axhline(
                threshold, color="orange", linestyle="--", label=f"thr={threshold}", alpha=0.8
            )

        fft_pre_plot = (
            FftPlotPipe(
                fft_plot_input,
                title="Threshold | FPGA",
                sample_rate=sample_rate,
                plot_kwargs={"alpha": 0.6, "color": "green"},
            )
            if fft_plot_input
            else None
        )

        super().__init__([pre_plot, fft_pre_plot, ThresholdPipe(threshold, seed), post_plot], seed)

    def reset(self) -> None:
        super().reset()
        if self._ax is not None:
            self._ax.axhline(self._threshold, color="orange", linestyle="--", label=f"thr={self._threshold}")
