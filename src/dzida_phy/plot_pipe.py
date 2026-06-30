
from dataclasses import dataclass
from typing import Any, Literal

from matplotlib.axes import Axes
from numpy import dtype, ndarray
import numpy as np

from dzida_phy.signal_pipe import SignalPipe
from dzida_phy.physical_units import Quantity

@dataclass
class PlotInput:
    ax: Axes
    indxs: tuple[int, int]

@dataclass
class PlotInputFactory:
    axs: list[Axes]
    indxs: tuple[int, int]

    def __post_init__(self):
        self.ax_nr = 0

    def reset(self):
        self.__post_init__()

    def __call__(self) -> Any:
        self.ax_nr += 1
        return PlotInput(self.axs[self.ax_nr], self.indxs)


def auto_scale_units(max_value: float, kind: Literal["time", "freq"] = "time") -> tuple[float, str]:
    """Pick a scale factor and unit label for a value span, for time or frequency axes."""
    if kind == "time":
        thresholds = [(1.0, 1.0, "s"), (1e-3, 1e3, "ms"), (1e-6, 1e6, "μs")]
        default = (1e9, "ns")
    else:
        thresholds = [(1e9, 1e-9, "GHz"), (1e6, 1e-6, "MHz"), (1e3, 1e-3, "kHz")]
        default = (1.0, "Hz")

    for threshold, scale, unit in thresholds:
        if max_value > threshold:
            return scale, unit
    return default


class CapturePlotPipe(SignalPipe):
    """Pass-through pipe that captures one chunk's signal slice and plots it on `ax`."""

    def __init__(
        self,
        plt_in: PlotInput,
        title: str | None = None,
        sample_rate: Quantity | None = None,
        plot_kwargs: dict | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(seed)
        self._chunk_index = 0
        self._plot_indexes = plt_in.indxs
        self.ax = plt_in.ax
        self.plot_kwargs = plot_kwargs or {}
        if title:
            self.ax.set_title(title)

        self.sample_rate = sample_rate

    def _get_time_axis(self, n_samples: int) -> tuple[np.ndarray, str]:
        """Generate time axis using Quantity. If sample_rate not set, return index axis."""
        if self.sample_rate is None:
            return np.arange(n_samples), "sample index"

        # Get time per sample from Quantity
        time_per_sample = self.sample_rate.to_s()  # seconds per sample
        time_values = np.arange(n_samples) * time_per_sample

        scale, unit = auto_scale_units(time_values[-1], kind="time")
        return time_values * scale, f"time ({unit})"

    def plot(self, signal_values: ndarray[tuple[Any, ...], dtype[Any]]) -> None:
        raise NotImplementedError()

    def process(self, signal: ndarray[tuple[Any, ...], dtype[Any]]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        if self._chunk_index == self._plot_indexes[0]:
            self.plot(signal[self._plot_indexes[1]])
        self._chunk_index += 1
        return signal

    def reset(self) -> None:
        self._chunk_index = 0
        self.ax.clear()


class PlotPipe(CapturePlotPipe):
    def __init__(
        self,
        plt_in: PlotInput,
        plot_type: Literal['plot', 'bar'] = 'plot',
        title: str | None = None,
        sample_rate: Quantity | None = None,
        plot_kwargs: dict | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(plt_in, title, sample_rate, plot_kwargs, seed)
        self.type = plot_type

    def plot(self, signal_values: ndarray[tuple[Any, ...], dtype[Any]]):
        x_axis, x_label = self._get_time_axis(len(signal_values))

        match self.type:
            case 'plot':
                self.ax.plot(x_axis, signal_values,
                             **{'label': f"Signal at {self._plot_indexes}", **self.plot_kwargs})
            case 'bar':
                width = (x_axis[1] - x_axis[0]) * 0.8 if len(x_axis) > 1 else 0.8
                self.ax.bar(x_axis, signal_values,
                            **{'width': width, 'label': f"Signal at {self._plot_indexes}", **self.plot_kwargs})
        self.ax.set_xlabel(x_label)
        self.ax.grid(True)