from dzida_phy.fft_plot_pipe import FftPlotPipe
from dzida_phy.filter_pipe import LowpassPipe_Timed
from dzida_phy.physical_units import Quantity, ns
from dzida_phy.plot_pipe import PlotInput, PlotPipe
from dzida_phy.sampling_pipe import UpSampler_Timed
from dzida_phy.signal_pipe import CompoundPipe


class DiodePipe(CompoundPipe):
    def __init__(
        self,
        resolution: Quantity,
        slot_width: Quantity,
        rise_time: Quantity,
        plot_input: PlotInput | None = None,
        fft_plot_input: PlotInput | None = None,
        seed: int = 42,
    ) -> None:
        self.upsampler = UpSampler_Timed(resolution, slot_width)
        lowpass = LowpassPipe_Timed(rise_time, resolution)
        self.rise_time = rise_time
        self.plotpipe = (
            PlotPipe(plot_input, title="Diode", sample_rate=resolution, seed=seed) if plot_input else None
        )
        self.fftplotpipe = (
            FftPlotPipe(fft_plot_input, title="Diode", sample_rate=resolution, seed=seed)
            if fft_plot_input
            else None
        )
        self.slot_width = slot_width
        super().__init__([self.upsampler, lowpass, self.plotpipe, self.fftplotpipe], seed)


class CVLL_350_9(DiodePipe):
    def __init__(
        self,
        resolution: Quantity,
        slot_width: Quantity,
        plot_input: PlotInput | None = None,
        fft_plot_input: PlotInput | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            resolution,
            slot_width,
            Quantity((1 * ns).to_hz(), display="time"),
            plot_input,
            fft_plot_input,
            seed,
        )

        title = (
            f"CVLL_350_9 | Laser Diode - P_out:25W, Rise time: {self.rise_time}, Slot time: {self.slot_width}"
        )
        if self.plotpipe is not None:
            self.plotpipe.ax.set_title(title)
        if self.fftplotpipe is not None:
            self.fftplotpipe.ax.set_title(title)
