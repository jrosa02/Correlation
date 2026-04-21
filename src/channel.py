import asyncio

import numpy as np

_fs_channel = None

def get_fs_channel(seed: int, snr_db: float) -> 'FSChannel':
    """Lazily initialize and return static FSChannel instance."""
    global _fs_channel
    if _fs_channel is None:
        _fs_channel = FSChannel(seed, snr_db)
    return _fs_channel

def add_awgn(signal: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    """Add AWGN at the given SNR (dB). sigma^2 = signal_power / SNR_linear."""
    snr = 10 ** (snr_db / 10)
    sigma2 = 0.5 / snr  # reference: pulse amplitude=1, 50% bit probability
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, np.sqrt(sigma2), size=signal.shape)
    return signal + noise

class FSChannel:
    def __init__(self, seed, snr_db:float) -> None:
        self.rng = np.random.default_rng(seed)
        self.snr_db = snr_db

    def add_awgn(self, signal: np.ndarray) -> np.ndarray:
        """Add AWGN at the given SNR (dB). sigma^2 = signal_power / SNR_linear."""
        snr = 10 ** (self.snr_db / 10)
        sigma2 = 0.5 / snr  # reference: pulse amplitude=1, 50% bit probability
        noise = self.rng.normal(0, np.sqrt(sigma2), size=signal.shape)
        return signal + noise

    async def process(self, in_queue: asyncio.Queue, out_queue: asyncio.Queue):
        """Consume BinPPMSymbol arrays, produce RealPPMSymbol equivalents."""
        while True:
            chunk = await in_queue.get()
            if chunk is None:
                await out_queue.put(None)
                break
            real_chunk = self.add_awgn(chunk)
            await out_queue.put(real_chunk)
