from dzida_phy.noise_pipe import AWGN
from dzida_phy.filter_pipe import LowpassPipe_Timed
from dzida_phy.signal_pipe import CompoundPipe
from dzida_phy.plot_pipe import PlotInput, PlotPipe
from dzida_phy.physical_units import Quantity
from dzida_phy.quantization_pipe import QuantizatorPipe

class Analog2DigitalPipe(CompoundPipe):
    def __init__(self, resolution: Quantity, band: Quantity, noise_power: float,
                 bits: int = 8, v_range: tuple[float, float] | None = None,
                 plot_input: PlotInput | None = None, seed: int = 42) -> None:
        lowpass  = LowpassPipe_Timed(band, resolution)
        noise    = AWGN(noise_power)
        self.quant    = QuantizatorPipe(bits, v_range, seed)
        self.plotpipe = PlotPipe(plot_input, title="ADC", sample_rate=resolution, seed=seed) if plot_input else None
        super().__init__([lowpass, noise, self.quant, self.plotpipe], seed)

class LTC6268(Analog2DigitalPipe):
    def __init__(self, resolution: Quantity, band: Quantity, noise_power: float, v_range: tuple[float, float] | None = None,
                 plot_input: PlotInput | None = None, seed: int = 42) -> None:
        super().__init__(resolution, band, noise_power, 8, v_range, plot_input, seed)
        if self.plotpipe is not None:
            title = (f"LTC6268 | ADC - Bits: {self.quant.bits}, Sampling: {resolution.set_repr('freq')}")
            self.plotpipe.ax.set_title(title)