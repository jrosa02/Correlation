import numpy as np


def word_error_rate(tx: np.ndarray, rx: np.ndarray) -> float:
    """Fraction of PPM symbols decoded incorrectly."""
    n = min(len(tx), len(rx))
    return float(np.mean(tx[:n] != rx[:n]))


def bit_error_rate(tx: np.ndarray, rx: np.ndarray, n_slots: int) -> float:
    """BER for PPM slot-index arrays. Counts differing bits in XOR of slot indices."""
    n = min(len(tx), len(rx))
    bits_per_symbol = int(np.log2(n_slots))
    xor = np.bitwise_xor(tx[:n].astype(np.uint64), rx[:n].astype(np.uint64))
    bit_errors = np.unpackbits(xor.view(np.uint8).reshape(n, 8), axis=1, bitorder="big").sum()
    return float(bit_errors / (n * bits_per_symbol))


def per_bit_error_rate(tx: np.ndarray, rx: np.ndarray, n_slots: int) -> np.ndarray:
    """Error probability per bit position within a symbol (LSB=index 0).

    Returns float array of shape (log2(n_slots),).
    """
    n = min(len(tx), len(rx))
    bits_per_symbol = int(np.log2(n_slots))
    xor = np.bitwise_xor(tx[:n].astype(np.uint64), rx[:n].astype(np.uint64))
    # bitorder='little' → column k = bit k of the uint64 value
    unpacked = np.unpackbits(xor.view(np.uint8).reshape(n, 8), axis=1, bitorder="little")
    counts: np.ndarray = unpacked[:, :bits_per_symbol].sum(axis=0).astype(np.int64)
    return counts / n
