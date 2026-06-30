from dzida_phy.fft_plot_pipe import FftPlotPipe
from dzida_phy.filter_pipe import LowpassPipe_Timed
from dzida_phy.noise_pipe import AWGN
from dzida_phy.physical_units import Quantity
from dzida_phy.plot_pipe import PlotInput, PlotPipe
from dzida_phy.quantization_pipe import QuantizatorPipe
from dzida_phy.signal_pipe import CompoundPipe


class Analog2DigitalPipe(CompoundPipe):
    def __init__(
        self,
        resolution: Quantity,
        band: Quantity,
        noise_power: float,
        bits: int = 8,
        v_range: tuple[float, float] | None = None,
        plot_input: PlotInput | None = None,
        fft_plot_input: PlotInput | None = None,
        seed: int = 42,
    ) -> None:
        lowpass = LowpassPipe_Timed(band, resolution)
        noise = AWGN(noise_power)
        self.quant = QuantizatorPipe(bits, v_range, seed)
        self.plotpipe = (
            PlotPipe(plot_input, title="ADC", sample_rate=resolution, seed=seed) if plot_input else None
        )
        self.fftplotpipe = (
            FftPlotPipe(fft_plot_input, title="ADC", sample_rate=resolution, seed=seed)
            if fft_plot_input
            else None
        )
        super().__init__([lowpass, noise, self.quant, self.plotpipe, self.fftplotpipe], seed)


class HMCAD1511(Analog2DigitalPipe):
    def __init__(
        self,
        resolution: Quantity,
        band: Quantity,
        v_range: tuple[float, float] | None = None,
        plot_input: PlotInput | None = None,
        fft_plot_input: PlotInput | None = None,
        seed: int = 42,
    ) -> None:

        SNR_dB = 49
        SNR_linear = 10 ** (SNR_dB / 10)

        super().__init__(resolution, band, 1 / SNR_linear, 8, v_range, plot_input, fft_plot_input, seed)
        title = f"HMCAD 1511 | ADC - Bits: {self.quant.bits}, Sampling: {resolution.set_repr('freq')}"
        if self.plotpipe is not None:
            self.plotpipe.ax.set_title(title)
        if self.fftplotpipe is not None:
            self.fftplotpipe.ax.set_title(title)
