from dzida_phy.noise_pipe import AWGN
from dzida_phy.sampling_pipe import UpSampler_Timed
from dzida_phy.filter_pipe import LowpassPipe_Timed
from dzida_phy.signal_pipe import CompoundPipe
from dzida_phy.plot_pipe import PlotPipe
from dzida_phy.physical_units import MHz, Quantity, kHz, ns
from matplotlib.axes import Axes

class Analog2DigitalPipe(CompoundPipe):
    def __init__(self, resolution: Quantity, band: Quantity, noise_power: float, ax: Axes|None = None, seed: int = 42) -> None:
        lowpass = LowpassPipe_Timed(band, resolution)
        noise   = AWGN(noise_power)
        plotpipe = PlotPipe(ax=ax, title=f"ADC", sample_rate=resolution, seed=seed) if ax else None
        super().__init__([lowpass, noise, plotpipe], seed)