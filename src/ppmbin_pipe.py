import asyncio
from typing import Any
import numpy as np

from src.signal_pipe import SignalPipe

class BinPPMSymbol(np.ndarray):
    def __new__(cls, size:int, index:int):
        """Create a new BinaryPPMSymbol"""
        obj = np.asarray(np.zeros(size, dtype=bool)).view(cls)
        obj[index] = True
        return obj

class BinPPMGen(SignalPipe):
    def __init__(self, indices: np.ndarray, chunk_size: int, n_slots: int, seed: int|None = None):
        super().__init__(seed)
        self.indices = indices
        self.chunk_size = chunk_size
        self.n_slots = n_slots
        self._chunk_idx = 0

    def random(self, n_slots: int) -> BinPPMSymbol:
        return BinPPMSymbol(n_slots, int(self.rng.integers(0, n_slots)))
    
    def from_indices(self, n_slots: int, indices: np.ndarray) -> np.ndarray:
        arr = np.zeros((len(indices), n_slots+1), dtype=bool)
        arr[np.arange(len(indices)), indices] = True
        return arr
    
    def random_array(self, n_symbols, n_slots):
        """Efficient - vectorized array creation with object wrapper"""
        indices = self.rng.integers(0, n_slots, size=n_symbols)

        return self.from_indices(n_slots, indices)

    def generate(self):
        if self._chunk_idx >= len(self.indices):
            return None
        chunk = self.indices[self._chunk_idx:self._chunk_idx + self.chunk_size]
        symbols = self.from_indices(self.n_slots, chunk)
        self._chunk_idx += self.chunk_size
        return symbols

    def process(self, signal: np.ndarray) -> np.ndarray:
        raise NotImplementedError("BinPPMGen is source-only")

    def consume(self, signal: np.ndarray):
        raise NotImplementedError("BinPPMGen is source-only")

def generate_bits(n_slots: int, seed: int | None = None) -> BinPPMSymbol:
    """Generate random binary PPM symbol."""
    rng = np.random.default_rng(seed)
    return BinPPMSymbol(n_slots, int(rng.integers(0, n_slots)))
