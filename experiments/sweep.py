from pathlib import Path
import numpy as np
from src.pulse_gen import generate_bits, make_tx_pulse, make_pulse_template, build_signal
from src.runner import run_trial
from src import plots


def run(p: dict) -> None:
    """BER vs SNR sweep over bandpass_bw_sweep and correlator_len_sweep."""
    snr_range = p["snr_range_db"]
    sp = p["single"]

    bits = generate_bits(p["n_bits"], min_gap=p["min_gap_bits"], seed=p["seed"])
    tx_pulse = make_tx_pulse(p["pulse_shape"], p["sps"])
    corr_template = make_pulse_template(p["pulse_shape"], p["sps"], neg_len=p.get("neg_len"))
    tx_signal = build_signal(bits, tx_pulse, p["sps"])

    out = Path(p["output_dir"]) / "sweep"

    bw_ber: dict[str, np.ndarray] = {}
    for bw in p["bandpass_bw_sweep"]:
        bers = [
            run_trial(bits, tx_signal, corr_template, snr, bw, sp["corr_len"], p).ber
            for snr in snr_range
        ]
        bw_ber[f"bw={bw}"] = np.array(bers)
    plots.plot_ber_curve(snr_range, bw_ber, str(out / "ber_vs_snr_bw.png"))

    len_ber: dict[str, np.ndarray] = {}
    for corr_len in p["correlator_len_sweep"]:
        bers = [
            run_trial(bits, tx_signal, corr_template, snr, sp["bandpass_bw"], corr_len, p).ber
            for snr in snr_range
        ]
        len_ber[f"corr_len={corr_len}"] = np.array(bers)
    plots.plot_ber_curve(snr_range, len_ber, str(out / "ber_vs_snr_corr_len.png"))

    print(f"Saved: {out}/ber_vs_snr_bw.png  {out}/ber_vs_snr_corr_len.png")
