import numpy as np
from numba import njit

from dzida_phy.signal_pipe import SignalPipe


@njit
def _batch_symbols_numba(indices: np.ndarray, n_slots: int) -> np.ndarray:
    """Numba-compiled batch symbol creation: O(n) vs O(n²) with fancy indexing."""
    n_symbols = len(indices)
    arr = np.zeros((n_symbols, n_slots), dtype=np.uint8)
    for i in range(n_symbols):
        arr[i, indices[i]] = 1
    return arr


class BinPPMSymbol(np.ndarray):
    def __new__(cls, size: int, index: int):
        """Create binary PPM symbol. Optimized: direct allocation, uint8 dtype."""
        obj = np.zeros(size, dtype=np.uint8).view(cls)
        obj[index] = 1
        return obj


class BinPPMGen(SignalPipe):
    def __init__(self, indices: np.ndarray, chunk_size: int, n_slots: int, seed: int = 42):
        super().__init__(seed)
        self.indices = indices
        self.chunk_size = chunk_size
        self.n_slots = n_slots
        self._chunk_idx = 0

    def random(self, n_slots: int) -> BinPPMSymbol:
        return BinPPMSymbol(n_slots, int(self.rng.integers(0, n_slots)))

    def from_indices(self, n_slots: int, indices: np.ndarray) -> np.ndarray:
        """Vectorized batch creation via numba. Use uint8 instead of bool."""
        return _batch_symbols_numba(indices.astype(np.int64), n_slots)

    def random_array(self, n_symbols, n_slots):
        """Efficient - vectorized array creation with object wrapper"""
        indices = self.rng.integers(0, n_slots, size=n_symbols)

        return self.from_indices(n_slots, indices)

    def generate(self):
        if self._chunk_idx >= len(self.indices):
            return None
        chunk = self.indices[self._chunk_idx : self._chunk_idx + self.chunk_size]
        symbols = self.from_indices(self.n_slots, chunk)
        self._chunk_idx += self.chunk_size
        return symbols

    def reset(self) -> None:
        self._chunk_idx = 0


def generate_bits(n_slots: int, seed: int | None = None) -> BinPPMSymbol:
    """Generate random binary PPM symbol."""
    rng = np.random.default_rng(seed)
    return BinPPMSymbol(n_slots, int(rng.integers(0, n_slots)))
