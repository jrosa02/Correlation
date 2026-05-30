from sampling_pipe import UpSampler_Timed
from filter_pipe import LowpassPipe_Timed
from signal_pipe import CompoundPipe, SignalPipe
from src.physical_units import MHz, Quantity, kHz, ns

class DiodePipe(CompoundPipe):
    def __init__(self, resolution: Quantity, slot_width: Quantity, rise_time: Quantity, seed: int = 42) -> None:
        upsampler = UpSampler_Timed(resolution, slot_width)
        lowpass = LowpassPipe_Timed(rise_time, resolution)
        super().__init__([upsampler, lowpass], seed)

class CVLL_350_9(DiodePipe):
    def __init__(self, resolution: Quantity, slot_width: Quantity, seed: int = 42) -> None:
        super().__init__(resolution, slot_width, 1*ns, seed)
