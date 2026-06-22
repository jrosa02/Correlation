from dzida_phy.models.model1_time import Model1
from dzida_phy.physical_units import MHz, kHz, ns

model = Model1(
    snr=0.6,
    threshold=0.3,
    sample_rate=1600 * MHz,
    slot_rate=20 * ns,
    bandpass_low=100 * kHz,     
    bandpass_high=50 * MHz,    
    chunk_size=48,
    ppm_rank=1<<4,
    n_symbols=1<<12,
    seed=42,
    plotting=True,
)

result = model.run()

print(f"WER: {result.wer:.4f}")
print(f"BER: {result.ber:.6f}")
print(f"Per-bit BER: {result.per_bit_ber}")

model.save_plot("output/plot.png")
