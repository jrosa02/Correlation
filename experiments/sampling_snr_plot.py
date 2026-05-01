import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from src import AWGN, BinPPMGen, SignalPipeRunner, UpSampler, CorrPipe, BandpassPipe, ThresholdPipe, MaximumPipe, DecodeSink
from src.metrics import bit_error_rate, word_error_rate


seed = 42
chunk_size = 1024
ppm_rank = 1024
rng = np.random.default_rng(seed)
input_data = np.concatenate((rng.integers(0, ppm_rank, 2*chunk_size, dtype=int), [0, ppm_rank-1]))

noise_powers = np.logspace(-2, -0.3, 8)
sampling_rates = [4, 8, 16, 32, 64]
threshold = 0.3

ber_grid = np.zeros((len(sampling_rates), len(noise_powers)))
wer_grid = np.zeros((len(sampling_rates), len(noise_powers)))


def run_row(i, sr):
    for j, noise_pow in enumerate(noise_powers):
        runner = SignalPipeRunner(seed)
        runner.append(BinPPMGen(input_data, chunk_size, ppm_rank))
        runner.append(UpSampler(sr))
        runner.append(AWGN(noise_pow))
        runner.append(BandpassPipe(0.007, 0.7))
        runner.append(CorrPipe('rect', pulse_width=sr))
        runner.append(ThresholdPipe(threshold))
        runner.append(CorrPipe('triangle', pulse_width=sr))
        runner.append(MaximumPipe(rate=sr))
        decoder = DecodeSink(len(input_data), chunk_size, ppm_rank, sr)
        runner.append(decoder)
        runner.run()

        output_data = decoder.get_data
        ber_grid[i, j] = bit_error_rate(input_data, output_data, ppm_rank)
        wer_grid[i, j] = word_error_rate(input_data, output_data)
        print(f"sr={sr:2d}  noise={noise_pow:.3e}  BER={ber_grid[i,j]:.4f}  WER={wer_grid[i,j]:.4f}")

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
