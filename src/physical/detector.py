from noise_pipe import AWGN
import numpy as np
from sampling_pipe import UpSampler_Timed
from filter_pipe import LowpassPipe_Timed
from signal_pipe import CompoundPipe, SignalPipe
from src.physical_units import MHz, Quantity, kHz, ns

class DetectorPipe(CompoundPipe):
    def __init__(self, resolution: Quantity, band: Quantity, noise_power: float, seed: int = 42) -> None:
        lowpass = LowpassPipe_Timed(band, resolution)
        noise   = AWGN(noise_power)
        super().__init__([lowpass, noise], seed)

class DET08CL(DetectorPipe):
    def __init__(self, resolution: Quantity, band: Quantity, signal_power: float, seed: int = 42) -> None:
        NEP = lambda band_hz: np.float128(2*1e-15) * np.sqrt(band_hz, dtype=np.float128)
        super().__init__(resolution, band, signal_power/float(NEP(band)), seed)