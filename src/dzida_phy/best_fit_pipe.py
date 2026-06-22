from typing import Any

import numpy as np
from numba import njit, prange

from dzida_phy.physical_units import Quantity
from dzida_phy.signal_pipe import SignalPipe


@njit(parallel=True, cache=True, fastmath=True)
def _find_best_spike(signal, target_len):
    n_rows, n_cols = signal.shape
    offsets = np.zeros(n_rows, dtype=np.int64)
    for r in prange(n_rows):
        best_pos = 0
        best_diff = n_cols + 1
        in_run = False
        run_start = 0
        for c in range(n_cols):
            if signal[r, c] and not in_run:
                in_run = True
                run_start = c
            elif not signal[r, c] and in_run:
                in_run = False
                run_len = c - run_start
                diff = abs(run_len - target_len)
                if diff < best_diff:
                    best_diff = diff
                    best_pos = run_start + run_len // 2
        if in_run:
            run_len = n_cols - run_start
            diff = abs(run_len - target_len)
            if diff < best_diff:
                best_pos = run_start + run_len // 2
        offsets[r] = best_pos
    return offsets


class BestFitPipe_Simple(SignalPipe):
    def __init__(self, rate: int, seed: int = 42) -> None:
        super().__init__(seed)
        self.rate = rate

    def process(self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> np.ndarray:
        offsets = _find_best_spike(signal, self.rate)
        n_rows, n_cols = signal.shape
        out = np.zeros((n_rows, n_cols), dtype=np.float64)
        out[np.arange(n_rows), offsets] = 1.0
        return out

    def reset(self) -> None:
        pass


class BestFitPipe_Timed(BestFitPipe_Simple):
    def __init__(
        self,
        sample_rate: Quantity,
        slot_rate: Quantity,
        seed: int = 42,
    ) -> None:
        super().__init__(
            rate=round(sample_rate.to_hz() / slot_rate.to_hz()),
            seed=seed,
        )
