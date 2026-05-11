import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from src.models.model1 import Model1


seed = 42
chunk_size = 1024
ppm_rank = 1024

noise_powers = np.logspace(-2, -0.3, 8)
sampling_rates = [4, 8, 16, 32, 64]

ber_grid = np.zeros((len(sampling_rates), len(noise_powers)))
wer_grid = np.zeros((len(sampling_rates), len(noise_powers)))


def run_row(i, sr):
    for j, noise_pow in enumerate(noise_powers):
        model = Model1(
            snr=noise_pow,
            sampling_rate=sr,
            chunk_size=chunk_size,
            ppm_rank=ppm_rank,
            n_symbols=2 * chunk_size,
            seed=seed,
        )
        result = model.run()
        ber_grid[i, j] = result.ber
        wer_grid[i, j] = result.wer
        print(f"sr={sr:2d}  noise={noise_pow:.3e}  BER={result.ber:.4f}  WER={result.wer:.4f}")

try:
    with ThreadPoolExecutor(max_workers=len(sampling_rates)) as executor:
        futures = {executor.submit(run_row, i, sr): sr for i, sr in enumerate(sampling_rates)}
        for f in as_completed(futures):
            f.result()
except KeyboardInterrupt:
    print("KeyboardInterrupt, saving plot")

os.makedirs('output', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.tight_layout(h_pad=2, w_pad=4)

X, Y = np.meshgrid(noise_powers, sampling_rates)
all_pos = np.concatenate([ber_grid[ber_grid > 0], wer_grid[wer_grid > 0]])
floor = float(all_pos.min()) if len(all_pos) else 1e-7

for ax, grid, title in zip(axes, [ber_grid, wer_grid], ['BER', 'WER']):
    clipped = np.clip(grid, floor, 1.0)
    im = ax.pcolormesh(X, Y, clipped, norm=LogNorm(vmin=floor, vmax=1.0), cmap='viridis')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(noise_powers)
    ax.set_xticklabels([f'{v:.0e}' for v in noise_powers], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(sampling_rates)
    ax.set_yticklabels(sampling_rates)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel('Noise power')
    ax.set_ylabel('Sampling rate')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=title)

fig.savefig('output/sampling_snr_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved output/sampling_snr_heatmap.png")
