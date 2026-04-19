import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import erfc


def _save(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_waveforms(
    stages: dict[str, np.ndarray],
    sps: int,
    save_path: str,
    params: dict | None = None,
    corr_threshold: float | None = None,
) -> None:
    """Stack time-domain waveforms for each pipeline stage."""
    n = len(stages)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    total_samples = max(len(s) for s in stages.values())
    for ax, (label, sig) in zip(axes, stages.items()):
        if len(sig) * sps == total_samples:
            # bit-rate signal: draw one bar per bit slot aligned with sample-rate x-axis
            centers = np.arange(len(sig)) + 0.5
            ax.bar(centers, sig, width=1.0, align="center", color="C0", alpha=0.7)
            ax.set_xlim(0, total_samples / sps)
        else:
            t = np.arange(len(sig)) / sps
            ax.plot(t, sig, lw=0.8)
            if corr_threshold is not None and "Correlator" in label:
                ax.axhline(corr_threshold, color="r", lw=1.0, ls="--", label=f"threshold={corr_threshold:.3f}")
                ax.legend(fontsize=8, loc="upper right")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [symbols]")

    title = "Pipeline waveforms"
    if params:
        keys = ["pulse_shape", "duty_cycle", "min_gap_bits", "sps", "n_bits", "f_center", "seed"]
        param_str = "  |  ".join(f"{k}={params[k]}" for k in keys if k in params)
        title = f"Pipeline waveforms\n{param_str}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)


def plot_templates(
    tx_pulse: np.ndarray,
    corr_template: np.ndarray,
    sps: int,
    save_path: str,
    params: dict | None = None,
) -> None:
    """Side-by-side TX pulse and correlation template, one sample per index."""
    fig, (ax_tx, ax_corr) = plt.subplots(1, 2, figsize=(10, 3))
    t = np.arange(len(tx_pulse))

    ax_tx.stem(t, tx_pulse, markerfmt="C0o", linefmt="C0-", basefmt="k-")
    ax_tx.set_title("TX pulse (unipolar)")
    ax_tx.set_xlabel("Sample")
    ax_tx.axhline(0, color="k", lw=0.5)
    ax_tx.grid(True, alpha=0.3)

    t_c = np.arange(len(corr_template))
    ax_corr.stem(t_c, corr_template, markerfmt="C1o", linefmt="C1-", basefmt="k-")
    ax_corr.set_title("Correlation template (zero-mean)")
    ax_corr.set_xlabel("Sample")
    ax_corr.axhline(0, color="k", lw=0.5)
    ax_corr.grid(True, alpha=0.3)

    title = "Pulse templates"
    if params:
        keys = ["pulse_shape", "duty_cycle", "sps"]
        title += "  |  " + "  ".join(f"{k}={params[k]}" for k in keys if k in params)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)


def plot_ber_curve(
    snr_range: np.ndarray,
    ber_dict: dict[str, np.ndarray],
    save_path: str,
) -> None:
    """BER vs Eb/N0 for multiple parameter sets, with theoretical BPSK reference."""
    fig, ax = plt.subplots(figsize=(8, 5))
    snr_lin = 10 ** (snr_range / 10)
    ber_theory = 0.5 * erfc(np.sqrt(snr_lin))
    ax.semilogy(snr_range, ber_theory, "k--", lw=1.5, label="BPSK theory")
    for label, ber in ber_dict.items():
        ber_clipped = np.clip(ber, 1e-6, 1.0)
        ax.semilogy(snr_range, ber_clipped, marker=".", lw=1, label=label)
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.set_title("BER vs Eb/N0")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, save_path)


def plot_snr_gain(
    snr_in: np.ndarray,
    snr_out_dict: dict[str, np.ndarray],
    save_path: str,
) -> None:
    """Output SNR vs input SNR for multiple parameter sets."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snr_in, snr_in, "k--", lw=1.5, label="No gain (reference)")
    for label, snr_out in snr_out_dict.items():
        ax.plot(snr_in, snr_out, marker=".", lw=1, label=label)
    ax.set_xlabel("Input Eb/N0 (dB)")
    ax.set_ylabel("Output SNR (dB)")
    ax.set_title("SNR gain through correlation filter")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, save_path)
