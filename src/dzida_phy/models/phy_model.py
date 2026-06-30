import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from typing import cast

from dzida_phy import (
    BinPPMGen, RectCorrModule_Timed, DecodePlotSink_Timed,
    BestFitPipe_Timed, PlotPipe, FftPlotPipe, ThresholdModule, HighpassModule_Timed,
)
from dzida_phy.physical.adc import HMCAD1511
from dzida_phy.signal_pipe import CompoundPipe
from dzida_phy.physical.diode import CVLL_350_9
from dzida_phy.physical.detector import DET08CL
from dzida_phy.plot_pipe import PlotInputFactory
from dzida_phy.metrics import bit_error_rate, per_bit_error_rate, word_error_rate
from dzida_phy.models.model import ABCModel, ModelResult
from dzida_phy.physical_units import Quantity


class PhyModel(ABCModel):
    """Physical model with AGH satellite link budget.

    AGH (2026) 25W diode, 508mm RX aperture, 1550nm FSO link.
    Received power: ~18-21 nW nominal (913 km altitude, 5.2 mrad TX divergence)
    """

    def __init__(
        self,
        *,
        threshold: float = 0.3,
        sample_rate: Quantity,
        slot_rate: Quantity,
        bandpass_low: Quantity,
        bandpass_high: Quantity,
        chunk_size: int = 2048,
        ppm_rank: int = 1024,
        n_symbols: int | None = None,
        signal_power: float | None = None,
        seed: int = 42,
        plotting: bool = False,
    ) -> None:
        super().__init__(seed)
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.slot_rate = slot_rate
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.chunk_size = chunk_size
        self.ppm_rank = ppm_rank
        self.plotting = plotting

        # Received signal power (W). If None, use AGH nominal link budget.
        if signal_power is None:
            # AGH link budget: 25W TX, 5.2 mrad divergence, 508mm aperture
            # Path loss 913 km altitude with 5.81 dB pointing error
            # Result: ~21 nW (AGH nominal: ~18 nW)
            signal_power = self._agh_link_budget()
        self.signal_power = signal_power

        rng = np.random.default_rng(seed)
        n = n_symbols if n_symbols is not None else 8 * chunk_size
        self.input_data = np.concatenate(
            (rng.integers(0, ppm_rank, n, dtype=int), [0, ppm_rank - 1])
        )

        self.fig = None
        self.axes = None
        self.fig_fft = None
        self.axes_fft = None
        if plotting:
            self._init_figure()

        self.construct_pipeline()

    @staticmethod
    def _agh_link_budget() -> float:
        """AGH satellite link budget calculation (25W, 508mm RX, 913km altitude).

        Reference: AGH meeting notes 15.05.2026
        Returns: Received optical power in watts
        """
        # TX parameters
        p_tx_dbw = 13.98  # 25 W
        tx_gain_db = 56.13  # 5.2 mrad divergence (dB)

        # RX parameters
        rx_gain_db = 119.54  # 508 mm aperture, 39% obstruction (dB)

        # Link losses
        pointing_loss_db = -5.81  # 5.2 mrad pointing error
        path_loss_db = -260.62  # 913 km altitude

        # Total link budget
        total_db = p_tx_dbw + tx_gain_db + rx_gain_db + pointing_loss_db + path_loss_db
        p_rx_w = 10 ** (total_db / 10)

        return p_rx_w

    def _init_figure(self) -> None:
        self.fig, self.axes = plt.subplots(9, 1)
        self.fig.set_size_inches((16, 20))
        self.fig.tight_layout(h_pad=1, w_pad=1)

        self.fig_fft, self.axes_fft = plt.subplots(9, 1)
        self.fig_fft.set_size_inches((16, 20))
        self.fig_fft.tight_layout(h_pad=1, w_pad=1)

    def construct_pipeline(self) -> None:
        ax = self.axes
        ax_fft = self.axes_fft

        samples_per_slot = round(self.sample_rate.to_hz() / self.slot_rate.to_hz())

        # Build processing pipeline using physical components via CompoundPipe
        factory = PlotInputFactory(axs=cast(list[Axes], [None] + list(ax)) if ax is not None else [], indxs=(0, 14))
        factory_fft = PlotInputFactory(axs=cast(list[Axes], [None] + list(ax_fft)) if ax_fft is not None else [], indxs=(0, 14))

        def skip_fft_axis(title: str) -> None:
            """Consume one FFT axis slot for a symbol-rate stage with no meaningful FFT."""
            plt_in = factory_fft()
            if plt_in is not None:
                plt_in.ax.set_title(title)
                plt_in.ax.axis('off')

        pipes = [
            BinPPMGen(self.input_data, self.chunk_size, self.ppm_rank),
            PlotPipe(factory(), 'bar', title='PPM symbols'),
            skip_fft_axis('PPM symbols'),
            CVLL_350_9(self.sample_rate, self.slot_rate, plot_input=factory(), fft_plot_input=factory_fft()),
            DET08CL(self.sample_rate, Quantity(self.slot_rate.to_hz()*2), self.signal_power,
                    plot_input=factory(), fft_plot_input=factory_fft()),
            HighpassModule_Timed(Quantity(self.slot_rate.to_hz()/20), self.sample_rate,
                                 plot_input=factory(), fft_plot_input=factory_fft()),
            HMCAD1511(self.sample_rate, self.sample_rate, plot_input=factory(), fft_plot_input=factory_fft()),
            RectCorrModule_Timed(self.sample_rate, self.slot_rate,
                                plot_input=factory(), fft_plot_input=factory_fft()),
            ThresholdModule(self.threshold, self.sample_rate, plot_input=factory(), fft_plot_input=factory_fft()),
            BestFitPipe_Timed(self.sample_rate, self.slot_rate),
            PlotPipe(factory(), title='BestFitPipe | FPGA', sample_rate=self.sample_rate),
            FftPlotPipe(factory_fft(), title='BestFitPipe | FPGA', sample_rate=self.sample_rate),
            DecodePlotSink_Timed(
            len(self.input_data), self.chunk_size, self.ppm_rank,
            self.sample_rate, self.slot_rate,
            plot_input=factory(), fft_plot_input=factory_fft(),
        )
        ]

        self.decoder: DecodePlotSink_Timed = pipes[-1]

        self.runner.append(CompoundPipe(pipes))
        self._samples_per_slot = samples_per_slot

    def run(self) -> ModelResult:
        self.runner.run()

        if self.axes is not None and self.fig is not None:
            self.fig.tight_layout(h_pad=1, w_pad=1)
        if self.axes_fft is not None and self.fig_fft is not None:
            self.fig_fft.tight_layout(h_pad=1, w_pad=1)

        decoded = self.decoder.decoded_data
        return ModelResult(
            ber=bit_error_rate(self.input_data, decoded, self.ppm_rank),
            wer=word_error_rate(self.input_data, decoded),
            per_bit_ber=per_bit_error_rate(self.input_data, decoded, self.ppm_rank),
            decoded=decoded,
        )

    def reset(self) -> None:
        self.runner.reset()

    def save_plot(self, path: str, fft_path: str | None = None) -> None:
        if self.fig is None:
            raise RuntimeError("plotting=False, no figure to save")
        self.fig.savefig(path, dpi=800)

        if fft_path is not None:
            if self.fig_fft is None:
                raise RuntimeError("plotting=False, no FFT figure to save")
            self.fig_fft.savefig(fft_path, dpi=800)
