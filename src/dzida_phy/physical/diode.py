from dzida_phy.sampling_pipe import UpSampler_Timed
from dzida_phy.filter_pipe import LowpassPipe_Timed
from dzida_phy.plot_pipe import PlotPipe, PlotInput
from dzida_phy.signal_pipe import CompoundPipe, SignalPipe
from dzida_phy.physical_units import MHz, Quantity, kHz, ns

class DiodePipe(CompoundPipe):
    def __init__(self, resolution: Quantity, slot_width: Quantity, rise_time: Quantity, plot_input: PlotInput | None = None, seed: int = 42) -> None:
        self.upsampler = UpSampler_Timed(resolution, slot_width)
        lowpass = LowpassPipe_Timed(rise_time, resolution)
        self.rise_time = rise_time
        self.plotpipe = PlotPipe(plot_input, title="Diode", sample_rate=resolution, seed=seed) if plot_input else None
        self.slot_width = slot_width
        super().__init__([self.upsampler, lowpass, self.plotpipe], seed)

class CVLL_350_9(DiodePipe):
    def __init__(self, resolution: Quantity, slot_width: Quantity, plot_input: PlotInput | None = None, seed: int = 42) -> None:
        super().__init__(resolution, slot_width, Quantity((1*ns).to_hz(), display="time"), plot_input, seed)

        if self.plotpipe is not None:
            title = (f"CVLL_350_9 | Laser Diode - P_out:25W, Rise time: {self.rise_time}, Slot time: {self.slot_width}")
            self.plotpipe.ax.set_title(title)
