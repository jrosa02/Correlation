import abc
from dataclasses import dataclass

import numpy as np

from src.runner_pipe import SignalPipeRunner


@dataclass
class ModelResult:
    ber: float
    wer: float
    per_bit_ber: np.ndarray
    decoded: np.ndarray


class ABCModel(abc.ABC):
    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self.seed = seed
        self.runner = SignalPipeRunner(seed)

    @abc.abstractmethod
    def construct_pipeline(self) -> None:
        pass

    @abc.abstractmethod
    def run(self) -> ModelResult:
        pass

    def reset(self) -> None:
        self.runner.reset()
