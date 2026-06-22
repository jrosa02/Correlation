import numpy as np
from matplotlib import pyplot as plt

from dzida_phy import (
    AWGN, BandpassPipe_Timed, BinPPMGen, CorrPipe_Timed, DecodeSink_Timed,
    BestFitPipe_Timed, PlotPipe, SignalPipeRunner, ThresholdPipe, UpSampler_Timed,
)
from dzida_phy.metrics import bit_error_rate, per_bit_error_rate, word_error_rate
from dzida_phy.models.model import ABCModel, ModelResult
from dzida_phy.physical_units import Quantity


class Model1(ABCModel):
    def __init__(
        self,
        *,
        snr: float = 0.2,
        threshold: float = 0.3,
        sample_rate: Quantity,
        slot_rate: Quantity,
        bandpass_low: Quantity,
        bandpass_high: Quantity,
        chunk_size: int = 2048,
        ppm_rank: int = 1024,
        n_symbols: int | None = None,
        seed: int = 42,
        plotting: bool = False,
    ) -> None:
        super().__init__(seed)
        self.snr = snr
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.slot_rate = slot_rate
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.chunk_size = chunk_size
        self.ppm_rank = ppm_rank
        self.plotting = plotting

        rng = np.random.default_rng(seed)
        n = n_symbols if n_symbols is not None else 8 * chunk_size
        self.input_data = np.concatenate(
            (rng.integers(0, ppm_rank, n, dtype=int), [0, ppm_rank - 1])
        )

        self.fig = None
        self.axes = None
        if plotting:
            self._init_figure()

        self.construct_pipeline()

    def _init_figure(self) -> None:
        self.fig, self.axes = plt.subplots(7, 1)
        self.fig.set_size_inches((8, 10))
        self.fig.tight_layout(h_pad=1, w_pad=1)

    def construct_pipeline(self) -> None:
        plot_indexes = (0, 14)
        ax = self.axes

        self.decoder = DecodeSink_Timed(
            len(self.input_data), self.chunk_size, self.ppm_rank,
            self.sample_rate, self.slot_rate,
        )

        p = ax is not None
        samples_per_slot = round(self.sample_rate.to_hz() / self.slot_rate.to_hz())

        self.runner.append(BinPPMGen(self.input_data, self.chunk_size, self.ppm_rank))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[0], 'bar', title='PPM symbols'))
        self.runner.append(UpSampler_Timed(self.sample_rate, self.slot_rate))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[1], title='Upsampled'))
        self.runner.append(AWGN(self.snr))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[2], title='AWGN'))
        self.runner.append(BandpassPipe_Timed(self.bandpass_low, self.bandpass_high, self.sample_rate))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[3], title='Bandpass'))
        self.runner.append(CorrPipe_Timed(self.sample_rate, self.slot_rate))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[4], title='Rect correlator'))
        self.runner.append(ThresholdPipe(self.threshold))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[5], title='Threshold'))
        self.runner.append(BestFitPipe_Timed(self.sample_rate, self.slot_rate))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[6], title='BestFitPipe'))
        self.runner.append(self.decoder)

        self._samples_per_slot = samples_per_slot

    def run(self) -> ModelResult:
        self.runner.run()

        if self.axes is not None and self.fig is not None:
            self.axes[4].plot(
                [0, self.ppm_rank * self._samples_per_slot],
                [self.threshold, self.threshold],
                "--r",
            )
            self.fig.tight_layout(h_pad=1, w_pad=1)

        decoded = self.decoder.get_data
        return ModelResult(
            ber=bit_error_rate(self.input_data, decoded, self.ppm_rank),
            wer=word_error_rate(self.input_data, decoded),
            per_bit_ber=per_bit_error_rate(self.input_data, decoded, self.ppm_rank),
            decoded=decoded,
        )

    def reset(self) -> None:
        self.runner.reset()

    def save_plot(self, path: str) -> None:
        if self.fig is None:
            raise RuntimeError("plotting=False, no figure to save")
        self.fig.savefig(path)
