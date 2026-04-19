from pathlib import Path
from src.pulse_gen import generate_bits, make_tx_pulse, make_pulse_template, build_signal
from src.runner import run_trial
from src import plots


def run(p: dict) -> None:
    """Single pass: one fixed eb_n0_db, one bandpass_bw, one corr_len."""
    sp = p["single"]

    bits = generate_bits(p["n_bits"], min_gap=p["min_gap_bits"], seed=p["seed"])
    tx_pulse = make_tx_pulse(p["pulse_shape"], p["sps"], duty_cycle=p["duty_cycle"])
    corr_template = make_pulse_template(p["pulse_shape"], p["sps"], duty_cycle=p["duty_cycle"])
    tx_signal = build_signal(bits, tx_pulse, p["sps"])

    result = run_trial(
        bits, tx_signal, corr_template,
        sp["eb_n0_db"], sp["bandpass_bw"], sp["corr_len"], p,
        threshold=sp.get("threshold"),
    )
    result.stages = {"Bits (pre-upsample)": bits} | result.stages | {"Decoded bits": result.decisions}

    print(f"BER={result.ber:.4f}  SNR_out={result.snr_out:.2f} dB")

    out = Path(p["output_dir"]) / "single"
    plots.plot_templates(tx_pulse, corr_template, p["sps"], str(out / "templates.png"), params=p)
    plots.plot_waveforms(result.stages, p["sps"], str(out / "waveforms.png"), params=p, corr_threshold=result.corr_threshold)
    print(f"Saved: {out}/templates.png  {out}/waveforms.png")
