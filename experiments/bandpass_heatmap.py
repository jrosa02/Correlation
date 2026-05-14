import json
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

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


if __name__ == "__main__":
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
        with ProcessPoolExecutor(max_workers=min(len(tasks), os.cpu_count()//2)) as executor:
            futures = {executor.submit(run_cell, t): t for t in tasks}
            for f in as_completed(futures):
                i, j, k, ber, wer = f.result()
                ber_raw[i, j, k] = ber
                wer_raw[i, j, k] = wer
    except KeyboardInterrupt:
        print("KeyboardInterrupt, saving plot")

    # Normalize each SNR slice by its grid max so every noise level contributes equally
    ber_norm = np.zeros_like(ber_raw)
    wer_norm = np.zeros_like(wer_raw)
    for k in range(n_snr):
        ber_max = ber_raw[:, :, k].max()
        wer_max = wer_raw[:, :, k].max()
        if ber_max > 0:
            ber_norm[:, :, k] = ber_raw[:, :, k] / ber_max
        if wer_max > 0:
            wer_norm[:, :, k] = wer_raw[:, :, k] / wer_max

    ber_grid = ber_norm.mean(axis=2)
    wer_grid = wer_norm.mean(axis=2)

    os.makedirs('output', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.tight_layout(h_pad=2, w_pad=4)

    xi = np.arange(len(bandpass_lows))
    yi = np.arange(len(bandpass_highs))
    X, Y = np.meshgrid(xi, yi)
    all_pos = np.concatenate([ber_grid[ber_grid > 0], wer_grid[wer_grid > 0]])
    floor = float(all_pos.min()) if len(all_pos) else 1e-7

    for ax, grid, title in zip(axes, [ber_grid, wer_grid], ['Normalized BER', 'Normalized WER']):
        clipped = np.clip(grid, floor, 1.0)
        im = ax.pcolormesh(X, Y, clipped.T, norm=LogNorm(vmin=clipped.min(), vmax=clipped.max()), cmap='viridis')
        ax.set_xticks(xi)
        ax.set_xticklabels([f'{v:.0e}' for v in bandpass_lows], rotation=45, ha='right', fontsize=7)
        ax.set_yticks(yi)
        ax.set_yticklabels([f'1-{1-v:.0e}' for v in bandpass_highs])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel('bandpass_low')
        ax.set_ylabel('bandpass_high')
        ax.set_title(f'{title} (mean of {n_snr} per-SNR max-normalized slices)')
        fig.colorbar(im, ax=ax, label=title)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(f'output/bandpass_heatmap_{ts}.png', dpi=150, bbox_inches='tight')
    print(f"Saved output/bandpass_heatmap_{ts}.png")

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
        "normalized_ber": ber_grid.tolist(),
        "normalized_wer": wer_grid.tolist(),
    }
    with open(f'output/bandpass_heatmap_{ts}.json', 'w') as f:
        json.dump(results, f)
    print(f"Saved output/bandpass_heatmap_{ts}.json")
