import numpy as np

from dzida_phy.fft_plot_pipe import FftPlotPipe
from dzida_phy.filter_pipe import LowpassPipe_Timed
from dzida_phy.noise_pipe import AWGN, BrownianNoise
from dzida_phy.physical_units import Quantity
from dzida_phy.plot_pipe import PlotInput, PlotPipe
from dzida_phy.signal_pipe import CompoundPipe


class DetectorPipe(CompoundPipe):
    def __init__(
        self,
        resolution: Quantity,
        band: Quantity,
        noise_to_signal: float,
        brownian_power: float | None = None,
        plot_input: PlotInput | None = None,
        fft_plot_input: PlotInput | None = None,
        seed: int = 42,
    ) -> None:
        if not isinstance(band, Quantity) and hasattr(band, "__float__"):
            band = Quantity(float(band))

        lowpass = LowpassPipe_Timed(band, resolution)
        awgn = AWGN(noise_to_signal)
        brownian = BrownianNoise(brownian_power)
        self.plotpipe = (
            PlotPipe(plot_input, title="Detector", sample_rate=resolution, seed=seed) if plot_input else None
        )
        self.fftplotpipe = (
            FftPlotPipe(fft_plot_input, title="Detector", sample_rate=resolution, seed=seed)
            if fft_plot_input
            else None
        )
        super().__init__([lowpass, awgn, brownian, self.plotpipe, self.fftplotpipe], seed)


class DET08CL(DetectorPipe):
    """Thorlabs DET08CL InGaAs PIN detector with Johnson (thermal) noise.

    Biased photodetector with adjustable load resistance RL for SNR tuning.
    Johnson noise dominates over shot noise in this regime (~nW signals).

    Design equations (from AGH notes 15.05.2026):
        Bandwidth: f_BW = 1 / (2π * R_L * C_j)
        Johnson current: I_J = √(4*k_B*T*f_BW / R_L)
        Noise power: P_noise = (I_J / R)²  where R is responsivity
    """

    # Detector parameters from Thorlabs DET08CL datasheet (1550 nm)
    RESPONSIVITY = 0.90  # A/W @ 1550nm (peak)
    DARK_CURRENT = 1.5e-9  # A (typical @ 12V bias)
    JUNCTION_CAPACITANCE = 0.3e-12  # F (0.3 pF)

    # Physical constants
    BOLTZMANN_CONSTANT = 1.381e-23  # J/K
    TEMPERATURE = 300  # K (room temperature)

    # Thermal drift model: InGaAs dark current doubles per 20°C
    DARK_CURRENT_TEMP_COEFF = np.log(2) / 20  # per °C
    THERMAL_DRIFT_RATE = 10  # K/√sample — environmental temperature fluctuation rate

    def __init__(
        self,
        resolution: Quantity,
        band: Quantity,
        signal_power: float,
        load_resistance: float | None = None,
        plot_input: PlotInput | None = None,
        fft_plot_input: PlotInput | None = None,
        seed: int = 42,
    ) -> None:
        """
        Args:
            resolution: Sample rate (Quantity)
            band: Detector bandwidth. R_L derived via R_L = 1/(2π*f_BW*C_j)
            signal_power: Received optical power (W)
            load_resistance: Override R_L (ohms). If None, derived from band.
            plot_input: Optional PlotInput for visualization
            fft_plot_input: Optional PlotInput for FFT spectrum visualization
            seed: Random seed
        """
        if not isinstance(band, Quantity):
            band = Quantity(float(band))
        f_bw = band.to_hz()

        # Derive load resistance from bandwidth: R_L = 1 / (2π * f_BW * C_j)
        if load_resistance is None:
            load_resistance = 1.0 / (2 * np.pi * f_bw * self.JUNCTION_CAPACITANCE)

        # Johnson (thermal) noise current: I_J = √(4*k_B*T*f_BW / R_L)
        i_johnson = np.sqrt(4 * self.BOLTZMANN_CONSTANT * self.TEMPERATURE * f_bw / load_resistance)

        # Dark current shot noise: I_dark = √(2*q*I_dark*f_BW)
        i_dark = np.sqrt(2 * 1.602e-19 * self.DARK_CURRENT * f_bw)

        # Total noise current
        i_noise = np.sqrt(i_johnson**2 + i_dark**2)

        # Noise power: P_noise = I_noise / R  (AGH doc, NOT squared)
        noise_power = i_noise / self.RESPONSIVITY

        snr_linear = signal_power / noise_power if noise_power > 0 else 1e10
        snr_db = 10 * np.log10(snr_linear)
        noise_to_signal = noise_power / signal_power if signal_power > 0 else 1.0

        thermal_drift_amp = (
            (
                self.DARK_CURRENT
                * self.DARK_CURRENT_TEMP_COEFF
                / self.RESPONSIVITY
                * self.THERMAL_DRIFT_RATE
                / signal_power
            )
            if signal_power > 0
            else 0.0
        )
        thermal_brownian_power = thermal_drift_amp**2

        super().__init__(
            resolution, band, noise_to_signal, thermal_brownian_power, plot_input, fft_plot_input, seed
        )

        title = (
            f"DET08CL | Photodiode - (R_L={load_resistance / 1e3:.0f}kΩ), "
            f"P_in:{signal_power:.2e}W, f_BW:{f_bw / 1e6:.1f}MHz, "
            f"SNR:{snr_db:.3f}dB"
        )
        if self.plotpipe is not None:
            self.plotpipe.ax.set_title(title)
        if self.fftplotpipe is not None:
            self.fftplotpipe.ax.set_title(title)
