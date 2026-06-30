from dzida_phy.models.phy_model import PhyModel
from dzida_phy.physical_units import MHz, kHz, ns

model = PhyModel(
    threshold = 0.3,
    sample_rate = 0.5 * ns,
    slot_rate = 64 * ns,
    bandpass_low = 10 * kHz,
    bandpass_high = 1000 * MHz,
    chunk_size = 48,
    ppm_rank = 1<<10,
    n_symbols = 1<<10,
    seed = 42,
    plotting = True,
)

result = model.run()

print(f"WER: {result.wer:.4f}")
print(f"BER: {result.ber:.6f}")
print(f"Per-bit BER: {result.per_bit_ber}")

model.save_plot("output/plot.png", "output/fft.png")
