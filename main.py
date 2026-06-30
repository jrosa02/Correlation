import argparse
from pathlib import Path

import experiments.ber_curve
import experiments.minimize
import experiments.single
import experiments.sweep
import numpy as np
import yaml


def load_params(config_path: str, overrides: dict) -> dict:
    with Path(config_path).open() as f:
        p = yaml.safe_load(f)
    p["snr_range_db"] = np.arange(*p["snr_range_db"])
    # Apply scalar CLI overrides; single sub-dict overrides handled separately
    single_overrides = {
        k: v for k, v in overrides.items() if k in ("snr_db", "bandpass_bw", "corr_len") and v is not None
    }
    top_overrides = {k: v for k, v in overrides.items() if k not in single_overrides and v is not None}
    p.update(top_overrides)
    p.setdefault("single", {})
    p["single"].update(single_overrides)
    return p


def parse_args():
    parser = argparse.ArgumentParser(description="Correlation filter experiment")
    parser.add_argument("--config", default="config.yaml")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # shared overrides added to every subcommand
    def add_common(sub):
        sub.add_argument("--n_slots", type=int)
        sub.add_argument("--sps", type=int)
        sub.add_argument("--pulse_shape", choices=["rect", "gaussian", "raised_cosine", "delta"])
        sub.add_argument("--duty_cycle", type=float)
        sub.add_argument("--seed", type=int)
        sub.add_argument("--f_center", type=float)
        sub.add_argument("--output_dir")

    single = subparsers.add_parser("single", help="One pass with fixed parameters")
    add_common(single)
    single.add_argument("--snr_db", type=float, help="Override single.snr_db")
    single.add_argument("--bandpass_bw", type=float, help="Override single.bandpass_bw")
    single.add_argument("--corr_len", type=int, help="Override single.corr_len")
    single.add_argument("--threshold", type=float, default=None, help="Override single.threshold (None=auto)")

    sweep = subparsers.add_parser("sweep", help="Sweep parameter ranges")
    add_common(sweep)

    minimize = subparsers.add_parser("minimize", help="Optimise filter parameters")
    add_common(minimize)

    ber_curve = subparsers.add_parser("ber_curve", help="BER vs SNR curve at fixed params")
    add_common(ber_curve)

    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k not in ("mode", "config")}
    p = load_params(args.config, overrides)

    match args.mode:
        case "single":
            experiments.single.run(p)
        case "sweep":
            experiments.sweep.run(p)
        case "minimize":
            experiments.minimize.run(p)
        case "ber_curve":
            experiments.ber_curve.run(p)


if __name__ == "__main__":
    main()
