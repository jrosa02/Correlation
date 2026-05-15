import multiprocessing
import json
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm

from src.models.model1 import Model1


seed = 42
chunk_size = 1024
ppm_rank = 1024
sampling_rate = 32

bandpass_lows = np.logspace(-3, -1, 4)
bandpass_highs = 1-np.logspace(-5, -1, 5)
noise_powers = np.logspace(-0.6, -0.2, 3)


def run_cell(args):
    i, j, k, bp_low, bp_high, noise_pow = args
    model = Model1(
        snr=noise_pow,
        bandpass_low=bp_low,
        bandpass_high=bp_high,
        sampling_rate=sampling_rate,
        chunk_size=chunk_size,
        ppm_rank=ppm_rank,
        n_symbols=1*chunk_size,
        seed=seed,
    )
    result = model.run()
    print(f"bp_low={bp_low:.3f}  bp_high={bp_high:.2f}  noise={noise_pow:.3e}  BER={result.ber:.4f}  WER={result.wer:.4f}")
    return i, j, k, result.ber, result.wer


def _draw_row(fig, axes, data_2d, param_vals, param_name, noise_powers, raw_std, label, snr_colors):
    """Fill one row of 3 axes: heatmap | line plot | diagnostic bar."""
    ax_heat, ax_line, ax_diag = axes
    n_params = len(param_vals)
    n_snr = len(noise_powers)

    floor = data_2d[data_2d > 0].min() if (data_2d > 0).any() else 1e-6
    clipped = np.clip(data_2d, floor, 1.0)
    norm = PowerNorm(gamma=0.35, vmin=clipped.min(), vmax=clipped.max())

    xi = np.arange(n_snr + 1) - 0.5
    yi = np.arange(n_params + 1) - 0.5
    X, Y = np.meshgrid(xi, yi)
    im = ax_heat.pcolormesh(X, Y, clipped, norm=norm, cmap='viridis')
    ax_heat.set_xticks(np.arange(n_snr))
    ax_heat.set_xticklabels([f'{v:.3f}' for v in noise_powers], rotation=45, ha='right', fontsize=7)
    ax_heat.set_yticks(np.arange(n_params))
    ax_heat.set_yticklabels([f'{v:.1e}' for v in param_vals], fontsize=7)
    ax_heat.set_xlabel('noise power')
    ax_heat.set_ylabel(param_name)
    collapsed_axis = 'bandpass_high' if param_name == 'bandpass_low' else 'bandpass_low'
    ax_heat.set_title(f'{label} (avg over {collapsed_axis})')
    cb = fig.colorbar(im, ax=ax_heat, label=label)
    thresholds = [t for t in [0.01, 0.05, 0.1, 0.5] if clipped.min() < t < clipped.max()]
    if thresholds:
        cb.set_ticks(thresholds)
        cb.set_ticklabels([str(t) for t in thresholds])

    for k, (noise_pow, color) in enumerate(zip(noise_powers, snr_colors)):
        ax_line.plot(param_vals, data_2d[:, k],
                     marker='o', markersize=3, color=color,
                     label=f'noise={noise_pow:.3f}')
    ax_line.set_xscale('log')
    ax_line.set_yscale('log')
    ax_line.set_xlabel(param_name)
    ax_line.set_ylabel(label)
    ax_line.set_title(f'{label} vs {param_name} per noise level')
    ax_line.legend(fontsize=6)
    ax_line.grid(True, alpha=0.3)

    mean_std = raw_std.mean(axis=1)
    ax_diag.barh(np.arange(n_params), mean_std, color='steelblue', alpha=0.7)
    ax_diag.set_yticks(np.arange(n_params))
    ax_diag.set_yticklabels([f'{v:.1e}' for v in param_vals], fontsize=6)
    ax_diag.set_xlabel(f'mean std across\n{collapsed_axis}')
    ax_diag.set_title(f'{collapsed_axis}\nvariance (noise floor)')
    ax_diag.grid(True, alpha=0.3)


def plot_and_save(ber_raw, wer_raw, bandpass_lows, bandpass_highs, noise_powers, ts):
    # ber_raw shape: (n_lows, n_highs, n_snr)
    n_snr = len(noise_powers)

    ber_2d_low  = ber_raw.mean(axis=1)   # (n_lows,  n_snr)
    wer_2d_low  = wer_raw.mean(axis=1)
    ber_2d_high = ber_raw.mean(axis=0)   # (n_highs, n_snr)
    wer_2d_high = wer_raw.mean(axis=0)

    # std across the collapsed axis for the diagnostic panel
    ber_std_over_high = ber_raw.std(axis=1)   # (n_lows,  n_snr)
    wer_std_over_high = wer_raw.std(axis=1)
    ber_std_over_low  = ber_raw.std(axis=0)   # (n_highs, n_snr)
    wer_std_over_low  = wer_raw.std(axis=0)

    os.makedirs('output', exist_ok=True)
    snr_colors = plt.colormaps['plasma'](np.linspace(0.15, 0.85, n_snr))

    # 4 rows: BER/low, BER/high, WER/low, WER/high
    fig, axes = plt.subplots(4, 3, figsize=(18, 22))
    fig.suptitle(f'Bandpass sweep — {ts}', fontsize=13, y=1.002)

    row_configs = [
        (ber_2d_low,  bandpass_lows,  'bandpass_low',  ber_std_over_high, 'BER'),
        (ber_2d_high, bandpass_highs, 'bandpass_high', ber_std_over_low,  'BER'),
        (wer_2d_low,  bandpass_lows,  'bandpass_low',  wer_std_over_high, 'WER'),
        (wer_2d_high, bandpass_highs, 'bandpass_high', wer_std_over_low,  'WER'),
    ]
    for row_axes, (data_2d, param_vals, param_name, raw_std, label) in zip(axes, row_configs):
        _draw_row(fig, row_axes, data_2d, param_vals, param_name,
                  noise_powers, raw_std, label, snr_colors)

    fig.tight_layout()
    fname = f'output/bandpass_heatmap_{ts}.png'
    fig.savefig(fname, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--just_plot', metavar='JSON', default=None,
                        help='load existing JSON and re-plot without running simulation')
    args = parser.parse_args()

    if args.just_plot:
        with open(args.just_plot) as f:
            d = json.load(f)
        ber_raw = np.array(d['ber_raw'])
        wer_raw = np.array(d['wer_raw'])
        lows = np.array(d['params']['bandpass_lows'])
        highs = np.array(d['params']['bandpass_highs'])
        nps = np.array(d['params']['noise_powers'])
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_and_save(ber_raw, wer_raw, lows, highs, nps, ts)
        raise SystemExit(0)

    n_snr = len(noise_powers)
    ber_raw = np.zeros((len(bandpass_lows), len(bandpass_highs), n_snr))
    wer_raw = np.zeros_like(ber_raw)

    tasks = [
        (i, j, k, bp_low, bp_high, noise_pow)
        for i, bp_low in enumerate(bandpass_lows)
        for j, bp_high in enumerate(bandpass_highs)
        for k, noise_pow in enumerate(noise_powers)
    ]

    try:
        mp_ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=min(len(tasks), (os.cpu_count() or 2) // 2), mp_context=mp_ctx) as executor:
            futures = {executor.submit(run_cell, t): t for t in tasks}
            for f in as_completed(futures):
                i, j, k, ber, wer = f.result()
                ber_raw[i, j, k] = ber
                wer_raw[i, j, k] = wer
    except KeyboardInterrupt:
        print("KeyboardInterrupt, saving plot")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_and_save(ber_raw, wer_raw, bandpass_lows, bandpass_highs, noise_powers, ts)

    results = {
        "datetime": datetime.now().isoformat(),
        "params": {
            "seed": seed,
            "chunk_size": chunk_size,
            "ppm_rank": ppm_rank,
            "sampling_rate": sampling_rate,
            "bandpass_lows": bandpass_lows.tolist(),
            "bandpass_highs": bandpass_highs.tolist(),
            "noise_powers": noise_powers.tolist(),
        },
        "ber_raw": ber_raw.tolist(),
        "wer_raw": wer_raw.tolist(),
    }
    with open(f'output/bandpass_heatmap_{ts}.json', 'w') as f:
        json.dump(results, f)
    print(f"Saved output/bandpass_heatmap_{ts}.json")
