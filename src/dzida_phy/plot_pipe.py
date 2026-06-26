
from dataclasses import dataclass
from typing import Any, Literal

from matplotlib.axes import Axes
from numpy import dtype, ndarray
import numpy as np

from dzida_phy.signal_pipe import SignalPipe
from dzida_phy.physical_units import Quantity
from matplotlib import pyplot as plt

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


class PlotPipe(SignalPipe):
    def __init__(
        self,
        plt_in: PlotInput,
        plot_type: Literal['plot', 'bar'] = 'plot',
        title: str | None = None,
        sample_rate: Quantity | None = None,
        plot_kwargs: dict | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(seed)
        self._chunk_index = 0
        self._plot_indexes = plt_in.indxs
        self.ax = plt_in.ax
        self.type = plot_type
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

        # Auto-select best unit based on time span
        max_time = time_values[-1]
        if max_time > 1.0:
            scale, unit = 1.0, "s"
        elif max_time > 1e-3:
            scale, unit = 1e3, "ms"
        elif max_time > 1e-6:
            scale, unit = 1e6, "μs"
        else:
            scale, unit = 1e9, "ns"

        return time_values * scale, f"time ({unit})"

    def plot(self, signals: ndarray[tuple[Any, ...], dtype[Any]]):
        signal_index = self._plot_indexes[1]
        signal_values = signals[signal_index]
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

    def process(self, signal: ndarray[tuple[Any, ...], dtype[Any]]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        if self._chunk_index == self._plot_indexes[0]:
            self.plot(signal)
        self._chunk_index += 1
        return signal
    
    def reset(self) -> None:
        self._chunk_index = 0
        self.ax.clear()