

from typing import Any, Literal

from matplotlib.axes import Axes
from numpy import dtype, ndarray

from src.signal_pipe import SignalPipe
from matplotlib import pyplot as plt


class PlotPipe(SignalPipe):
    def __init__(self, indexes: tuple[int, int] = (0, 0), ax: Axes|None = None, plot_type:Literal['plot', 'bar'] = 'plot', title: str | None = None, seed: int = 42) -> None:
        super().__init__(seed)
        self._chunk_index = 0
        self._plot_indexes = indexes
        if ax:
            self.ax: Axes = ax
        else:
            _, self.ax = plt.subplots(figsize=(8, 4))
        self.type = plot_type
        if title:
            self.ax.set_title(title)
    
    def plot(self, signals: ndarray[tuple[Any, ...], dtype[Any]]):
        signal_index = self._plot_indexes[1]
        signal_values = signals[signal_index]
        match self.type:
            case 'plot':
                self.ax.plot(signal_values, label=f"Signal at {self._plot_indexes}")
            case 'bar':
                self.ax.bar([i for i in range(len(signal_values))], signal_values, label=f"Signal at {self._plot_indexes}")
        self.ax.grid(True)

    def process(self, signal: ndarray[tuple[Any, ...], dtype[Any]]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        if self._chunk_index == self._plot_indexes[0]:
            self.plot(signal)
        self._chunk_index += 1
        return signal
    
    def reset(self) -> None:
        self._chunk_index = 0
        self.ax.clear()