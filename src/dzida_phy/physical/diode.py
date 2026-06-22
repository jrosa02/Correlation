from dzida_phy.sampling_pipe import UpSampler_Timed
from dzida_phy.filter_pipe import LowpassPipe_Timed
from dzida_phy.plot_pipe import PlotPipe, PlotInput
from dzida_phy.signal_pipe import CompoundPipe, SignalPipe
from dzida_phy.physical_units import MHz, Quantity, kHz, ns

class DiodePipe(CompoundPipe):
    def __init__(self, resolution: Quantity, slot_width: Quantity, rise_time: Quantity, plot_input: PlotInput | None = None, seed: int = 42) -> None:
        upsampler = UpSampler_Timed(resolution, slot_width)
        lowpass = LowpassPipe_Timed(rise_time, resolution)
        plotpipe = PlotPipe(plot_input, title="Diode", sample_rate=resolution, seed=seed) if plot_input else None
        super().__init__([upsampler, lowpass, plotpipe], seed)

class CVLL_350_9(DiodePipe):
    def __init__(self, resolution: Quantity, slot_width: Quantity, plot_input: PlotInput | None = None, seed: int = 42) -> None:
        super().__init__(resolution, slot_width, 1*ns, plot_input, seed)
