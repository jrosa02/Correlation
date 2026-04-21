from pathlib import Path
import numpy as np
from src.ppmbin import generate_bits
from src.ppmsignal import make_tx_pulse, make_pulse_template, build_signal
from src.runner import run_trial
from src import plots


def run(p: dict) -> None:
    """BER vs SNR at fixed bandpass_bw and corr_len from single: section."""
    sp = p["single"]
    snr_range = p["snr_range_db"]

    bits = generate_bits(p["n_slots"], seed=p["seed"])
    tx_pulse = make_tx_pulse(p["pulse_shape"], p["sps"])
    corr_template = make_pulse_template(p["pulse_shape"], p["sps"], neg_len=p.get("neg_len"))
    tx_signal = build_signal(bits, tx_pulse, p["sps"])

    bers = [
        run_trial(bits, tx_signal, corr_template, snr, sp["bandpass_bw"], sp["corr_len"], p).ber
        for snr in snr_range
    ]

    out = Path(p["output_dir"]) / "ber_curve"
    plots.plot_ber_curve(snr_range, {"measured": np.array(bers)}, str(out / "ber_vs_snr.png"))
    print(f"Saved: {out}/ber_vs_snr.png")
