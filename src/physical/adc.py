from noise_pipe import AWGN
from sampling_pipe import UpSampler_Timed
from filter_pipe import LowpassPipe_Timed
from signal_pipe import CompoundPipe, SignalPipe
from src.physical_units import MHz, Quantity, kHz, ns

class Analog2DigitalPipe(CompoundPipe):
    def __init__(self, resolution: Quantity, band: Quantity, noise_power: float, seed: int = 42) -> None:
        lowpass = LowpassPipe_Timed(band, resolution)
        noise   = AWGN(noise_power)
        super().__init__([lowpass, noise], seed)