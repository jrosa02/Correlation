from dataclasses import dataclass
import numpy as np
from src.channel import add_awgn
from src.filters import bandpass_filter
from src.correlator import sliding_correlator
from src.metrics import compute_ber, compute_output_snr


@dataclass
class TrialResult:
    ber: float
    snr_out: float
    stages: dict        # stage_name -> np.ndarray waveform
    corr_threshold: float = 0.0
    decisions: np.ndarray | None = None


def run_trial(
    bits: np.ndarray,
    tx_signal: np.ndarray,
    corr_template: np.ndarray,
    eb_n0_db: float,
    bandpass_bw: float,
    corr_len: int,
    p: dict,
    threshold: float | None = None,
) -> TrialResult:
    noisy = add_awgn(tx_signal, eb_n0_db, p["sps"])
    filtered = bandpass_filter(noisy, p["fs"], p["f_center"], bandpass_bw)
    corr_out, decisions, threshold = sliding_correlator(filtered, corr_template, corr_len, p["sps"], threshold=threshold)
    n = min(len(bits), len(decisions))
    ber = compute_ber(bits[:n], decisions[:n])
    snr_out = compute_output_snr(tx_signal, filtered)
    stages = {
        "TX (clean)": tx_signal,
        f"After AWGN ({eb_n0_db:.0f} dB)": noisy,
        f"After BPF (BW={bandpass_bw})": filtered,
        f"After Correlator (len={corr_len})": corr_out,
    }
    return TrialResult(ber=ber, snr_out=snr_out, stages=stages, corr_threshold=threshold, decisions=decisions)
