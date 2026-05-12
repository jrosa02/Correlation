import numpy as np
from matplotlib import pyplot as plt

from src import (
    AWGN, BandpassPipe, BinPPMGen, CorrPipe, DecodeSink,
    MaximumPipe, PlotPipe, SignalPipeRunner, ThresholdPipe, UpSampler,
)
from src.metrics import bit_error_rate, per_bit_error_rate, word_error_rate
from src.models.model import ABCModel, ModelResult


class Model1(ABCModel):
    def __init__(
        self,
        *,
        snr: float = 0.2,
        threshold: float = 0.3,
        bandpass_low: float = 0.007,
        bandpass_high: float = 0.99,
        chunk_size: int = 2048,
        ppm_rank: int = 1024,
        sampling_rate: int = 32,
        n_symbols: int | None = None,
        seed: int = 42,
        plotting: bool = False,
    ) -> None:
        super().__init__(seed)
        self.snr = snr
        self.threshold = threshold
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.chunk_size = chunk_size
        self.ppm_rank = ppm_rank
        self.sampling_rate = sampling_rate
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
        self.fig, self.axes = plt.subplots(8, 1)
        self.fig.set_size_inches((8, 10))
        self.fig.tight_layout(h_pad=1, w_pad=1)

    def construct_pipeline(self) -> None:
        plot_indexes = (0, 14)
        ax = self.axes

        self.decoder = DecodeSink(
            len(self.input_data), self.chunk_size, self.ppm_rank, self.sampling_rate
        )

        p = ax is not None

        self.runner.append(BinPPMGen(self.input_data, self.chunk_size, self.ppm_rank))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[0], 'bar', title='PPM symbols'))
        self.runner.append(UpSampler(self.sampling_rate))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[1], title='Upsampled'))
        self.runner.append(AWGN(self.snr))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[2], title='AWGN'))
        self.runner.append(BandpassPipe(self.bandpass_low, self.bandpass_high))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[3], title='Bandpass'))
        self.runner.append(CorrPipe('rect', pulse_width=self.sampling_rate))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[4], title='Rect correlator'))
        self.runner.append(ThresholdPipe(self.threshold))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[5], title='Threshold'))
        self.runner.append(CorrPipe('triangle', pulse_width=self.sampling_rate))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[6], title='Triangle correlator'))
        self.runner.append(MaximumPipe(rate=self.sampling_rate))
        if p: self.runner.append(PlotPipe(plot_indexes, ax[7], title='Maximum'))
        self.runner.append(self.decoder)

    def run(self) -> ModelResult:
        self.runner.run()

        if self.axes is not None and self.fig is not None:
            self.axes[4].plot(
                [0, self.ppm_rank * self.sampling_rate],
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
