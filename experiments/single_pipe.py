
from typing import Any

from matplotlib.axes import Axes
import numpy as np
from matplotlib import pyplot as plt
from src import AWGN, PlotPipe, BinPPMGen, SignalPipeRunner, UpSampler, CorrPipe, BandpassPipe, ThresholdPipe, MaximumPipe, DecodeSink
from src.metrics import bit_error_rate, word_error_rate


fig, axes = plt.subplots(8, 1)
fig.set_size_inches((8, 10))
fig.tight_layout(h_pad=1, w_pad=1)

seed = 42
chunk_size = 2048
ppm_rank = 16
rng = np.random.default_rng(seed)
input_data = np.concatenate((rng.integers(0, ppm_rank, 8*chunk_size, dtype=int), [0, ppm_rank-1]))

sampling_rate = 8

snr = 0.3

threshold = 0.3

plot_indexes=(0, 14)


runner = SignalPipeRunner(seed)
runner.append(BinPPMGen(input_data, chunk_size, ppm_rank))
runner.append(PlotPipe(plot_indexes, axes[0], 'bar', title='PPM symbols'))
runner.append(UpSampler(sampling_rate))
runner.append(PlotPipe(plot_indexes, axes[1], title='Upsampled'))
runner.append(AWGN(snr))
runner.append(PlotPipe(plot_indexes, axes[2], title='AWGN'))
runner.append(BandpassPipe(0.007, 0.7))
runner.append(PlotPipe(plot_indexes, axes[3], title='Bandpass'))
runner.append(CorrPipe('rect', pulse_width=sampling_rate))
runner.append(PlotPipe(plot_indexes, axes[4], title='Rect correlator'))
runner.append(ThresholdPipe(threshold))
runner.append(PlotPipe(plot_indexes, axes[5], title='Threshold'))
runner.append(CorrPipe('triangle', pulse_width=sampling_rate))
runner.append(PlotPipe(plot_indexes, axes[6], title='Triangle correlator'))
runner.append(MaximumPipe(rate=sampling_rate))
runner.append(PlotPipe(plot_indexes, axes[7], title='Maximum'))
decoder = DecodeSink(len(input_data), chunk_size, ppm_rank, sampling_rate)
runner.append(decoder)
runner.run()

axes[4].plot([0, ppm_rank*sampling_rate], [threshold, threshold], "--r")

fig.savefig("plot.png")
output_data = decoder.get_data

print(runner)

print(input_data)
print(output_data)

print(word_error_rate(input_data, output_data))
print(bit_error_rate(input_data, output_data, ppm_rank))

