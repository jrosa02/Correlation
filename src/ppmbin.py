import asyncio
import numpy as np

class BinPPMSymbol(np.ndarray):
    def __new__(cls, size:int, index:int):
        """Create a new BinaryPPMSymbol"""
        obj = np.asarray(np.zeros(size, dtype=bool)).view(cls)
        obj[index] = True
        return obj

class BinPPMGen:
    def __init__(self, seed: int|None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def random(self, n_slots: int) -> BinPPMSymbol:
        return BinPPMSymbol(n_slots, int(self.rng.integers(0, n_slots)))
    
    def from_indices(self, n_slots:int, indices:np.ndarray):
        n_symbols = len(indices)
        arr = np.zeros((n_symbols, n_slots), dtype=bool)
        arr[np.arange(n_symbols), indices] = 1

        result = np.empty(n_symbols, dtype=object)
        for i in range(n_symbols):
            result[i] = arr[i].view(BinPPMSymbol)

        return result
    
    def random_array(self, n_symbols, n_slots):
        """Efficient - vectorized array creation with object wrapper"""
        indices = self.rng.integers(0, n_slots, size=n_symbols)

        return self.from_indices(n_slots, indices)

    async def produce(self, indices: np.ndarray, chunk_size:int , n_slots: int, queue: asyncio.Queue) -> None:
        """Consume array of indices, produce BinPPMSymbol objects into async queue in chunks."""
        for i in range(0, len(indices), chunk_size):
            chunk = indices[i:i + chunk_size]
            symbols = self.from_indices(n_slots, chunk)
            await queue.put(symbols)
        await queue.put(None)

def generate_bits(n_slots: int, seed: int | None = None) -> np.ndarray:
    """Generate sparse binary sequence."""
    ppm_datagen = BinPPMGen(seed)

    return ppm_datagen.random(n_slots)
