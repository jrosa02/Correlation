from src.models.model1 import Model1

model = Model1(
    snr=0.4,
    threshold=0.3,
    bandpass_low=0.007,
    bandpass_high=0.7,
    chunk_size=48,
    ppm_rank=16,
    sampling_rate=32,
    seed=42,
    plotting=True,
    fs = 10000
)

result = model.run()

print(f"WER: {result.wer:.4f}")
print(f"BER: {result.ber:.6f}")
print(f"Per-bit BER: {result.per_bit_ber}")

model.save_plot("output/plot.png")