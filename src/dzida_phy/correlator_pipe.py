from typing import Any, Literal

import numpy as np
from numpy import dtype, ndarray

from dzida_phy.fft_plot_pipe import FftPlotPipe, plot_fft_spectrum
from dzida_phy.native_optimized import correlate
from dzida_phy.physical_units import Quantity
from dzida_phy.plot_pipe import PlotInput, PlotPipe
from dzida_phy.signal_pipe import CompoundPipe, SignalPipe

_REF_TYPE = {"rect": 0, "triangle": 1}


def rect_reference(pulse_width: int) -> np.ndarray:
    """Recover the compiled rect matched-filter kernel via its impulse response.

    Runs the real corr_ext correlator on a unit impulse rather than re-deriving the
    kernel formula in Python, so this can't drift from the compiled C constants.
    """
    ref_len = 2 * pulse_width
    n = 4 * ref_len  # margin so the impulse response stays clear of edge-padding
    sig = np.zeros((1, n), dtype=np.float64)
    sig[0, n // 2] = 1.0
    out = np.empty_like(sig)
    correlate(sig, _REF_TYPE["rect"], pulse_width, out)
    row = out[0] * ref_len
    nz = np.flatnonzero(np.abs(row) > 1e-15)
    return row[nz[0] : nz[-1] + 1]


class CorrPipe_Simple(SignalPipe):
    def __init__(
        self, ref_type: Literal["rect", "triangle"] = "rect", pulse_width: int = 4, seed: int = 42
    ) -> None:
        super().__init__(seed)
        self.ref_type = ref_type
        self.pulse_width = pulse_width

    def process(self, signal: ndarray[tuple[Any, ...], dtype[Any]]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        rows = signal.reshape(-1, signal.shape[-1]) if signal.ndim > 1 else signal[np.newaxis]
        rows = np.ascontiguousarray(rows, dtype=np.float64)
        out = np.empty_like(rows)
        correlate(rows, _REF_TYPE[self.ref_type], self.pulse_width, out)
        return out if signal.ndim > 1 else out[0]

    def reset(self) -> None:
        pass


class CorrPipe_Timed(CorrPipe_Simple):
    def __init__(
        self,
        sample_rate: Quantity,
        slot_rate: Quantity,
        ref_type: Literal["rect", "triangle"] = "rect",
        seed: int = 42,
    ) -> None:
        super().__init__(
            ref_type=ref_type,
            pulse_width=round(sample_rate.to_hz() / slot_rate.to_hz()),
            seed=seed,
        )


class RectCorrModule_Timed(CompoundPipe):
    """Rect correlator + time/FFT plotting, with the reference template's FFT
    overlaid on the same axes as the correlator output's FFT."""

    def __init__(
        self,
        sample_rate: Quantity,
        slot_rate: Quantity,
        plot_input: PlotInput | None = None,
        fft_plot_input: PlotInput | None = None,
        seed: int = 42,
    ) -> None:
        corr = CorrPipe_Timed(sample_rate, slot_rate, seed=seed)
        plotpipe = (
            PlotPipe(plot_input, title="Rect correlator | FPGA", sample_rate=sample_rate)
            if plot_input
            else None
        )

        fftplotpipe = None
        if fft_plot_input is not None:
            fftplotpipe = FftPlotPipe(
                fft_plot_input,
                title="Rect correlator | FPGA",
                sample_rate=sample_rate,
                plot_kwargs={"label": "Correlator output", "zorder": 1},
            )
            ref = rect_reference(corr.pulse_width)
            plot_fft_spectrum(
                fftplotpipe.ax,
                ref,
                sample_rate,
                plot_kwargs={"label": "Reference", "linestyle": "--", "zorder": 2},
            )
            fftplotpipe.ax.legend()

        super().__init__([corr, plotpipe, fftplotpipe], seed)
