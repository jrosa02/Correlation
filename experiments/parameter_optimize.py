import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt

from src import AWGN, BinPPMGen, SignalPipeRunner, UpSampler, CorrPipe, BandpassPipe, ThresholdPipe, BestFitPipe, DecodeSink
from dzida_phy.metrics import bit_error_rate, word_error_rate

PPM_RANK = 1024
CHUNK_SIZE = 64
SAMPLING_RATE = 16
N_SYMBOLS = 64
N_TRIALS = 10
SEED = 42

def cost(x):
    threshold = float(x[0])
    low_filt = np.min([x[1], x[2]])
    high_filt = np.max([x[1], x[2]])
    errors = []
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(trial)
        input_data = rng.integers(0, PPM_RANK, N_SYMBOLS, dtype=int)

        runner = SignalPipeRunner(trial)
        runner.append(BinPPMGen(input_data, CHUNK_SIZE, PPM_RANK))
        runner.append(UpSampler(SAMPLING_RATE))
        snr = 0.3*float(trial)/(N_TRIALS)
        runner.append(AWGN(snr))
        runner.append(BandpassPipe(low_filt, high_filt))
        runner.append(CorrPipe('rect', pulse_width=SAMPLING_RATE))
        runner.append(ThresholdPipe(threshold))
        runner.append(BestFitPipe(rate=SAMPLING_RATE))
        decoder = DecodeSink(len(input_data), CHUNK_SIZE, PPM_RANK, SAMPLING_RATE)
        runner.append(decoder)
        runner.run()

        errors.append(bit_error_rate(input_data, decoder.get_data, PPM_RANK))
        errors.append(word_error_rate(input_data, decoder.get_data))

    return 100*float(np.mean(errors))


history: list[dict] = []

def callback(intermediate_result: scipy.optimize.OptimizeResult):
    history.append({
        'x': intermediate_result.x.copy(),
        'fun': intermediate_result.fun,
    })
    print(f"iter {len(history):3d} | cost={intermediate_result.fun:.4f} | x={np.round(intermediate_result.x, 4)}")

x0 = [0.3, 0.001, 0.8]

initial_simplex = np.array([
    [0.1, 0.001, 0.5],
    [0.5, 0.001, 0.9],
    [0.3, 0.3,   0.7],
    [0.8, 0.1,   0.999],
])

result = scipy.optimize.minimize(
    cost, x0, method='Nelder-Mead',
    bounds=[(0, 1), (0.001, 0.999), (0.001, 0.999)],
    options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 200, 'initial_simplex': initial_simplex},
    callback=callback,
)
print(result)

np.save("output/optimize_history.npy", np.array([{"x": h["x"], "fun": h["fun"]} for h in history], dtype=object))

iters = np.arange(1, len(history) + 1)
costs = [h['fun'] for h in history]
params = np.array([h['x'] for h in history])

fig, axes = plt.subplots(2, 1, figsize=(8, 6))

axes[0].plot(iters, costs, marker='o', markersize=3)
axes[0].set_ylabel("cost")
axes[0].set_title("Convergence")

axes[1].plot(iters, params[:, 0], label="threshold")
axes[1].plot(iters, params[:, 1], label="low_filt")
axes[1].plot(iters, params[:, 2], label="high_filt")
axes[1].set_ylabel("parameter value")
axes[1].set_xlabel("iteration")
axes[1].legend()

fig.tight_layout()
fig.savefig("output/optimize_convergence.png")
print("Saved output/optimize_convergence.png")
