from typing import Any

from numpy import dtype, ndarray
import numpy as np

from dzida_phy.physical_units import Quantity
from dzida_phy.plot_pipe import CapturePlotPipe, PlotInput, auto_scale_units


def plot_fft_spectrum(
    ax,
    signal_values: ndarray[tuple[Any, ...], dtype[Any]],
    sample_rate: Quantity,
    keep_fraction: float = 0.1,
    plot_kwargs: dict | None = None,
) -> None:
    """Compute and plot the FFT magnitude spectrum of `signal_values` onto `ax`,
    truncated to the lowest `keep_fraction` of frequency bins."""
    spectrum = np.fft.rfft(np.asarray(signal_values, dtype=np.float64))
    freqs = np.fft.rfftfreq(len(signal_values), d=sample_rate.to_s())

    n_keep = max(1, round(len(freqs) * keep_fraction))
    freqs = freqs[:n_keep]
    spectrum = spectrum[:n_keep]

    scale, unit = auto_scale_units(freqs[-1], kind="freq")
    ax.plot(freqs * scale, np.abs(spectrum), **(plot_kwargs or {}))
    ax.set_xlabel(f"frequency ({unit})")
    ax.set_ylabel("magnitude")
    ax.grid(True)


class FftPlotPipe(CapturePlotPipe):
    """Pass-through pipe that plots the FFT magnitude spectrum of a captured chunk."""

    def __init__(
        self,
        plt_in: PlotInput,
        title: str | None = None,
        sample_rate: Quantity | None = None,
        plot_kwargs: dict | None = None,
        seed: int = 42,
    ) -> None:
        assert sample_rate is not None, ValueError("FftPlotPipe requires sample_rate to compute a frequency axis")
        super().__init__(plt_in, title, sample_rate, plot_kwargs, seed)

    def plot(self, signal_values: ndarray[tuple[Any, ...], dtype[Any]]):
        assert self.sample_rate is not None
        plot_kwargs = {'label': f"Signal at {self._plot_indexes}", **self.plot_kwargs}
        plot_fft_spectrum(self.ax, signal_values, self.sample_rate, plot_kwargs=plot_kwargs)
