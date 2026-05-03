import asyncio
from typing import Any, Coroutine, Literal

import numpy as np

from src.signal_pipe import SignalPipe


class SignalPipeRunner():
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.pipe_elements: list[SignalPipe] = list()

    def append(self, pipe_element: SignalPipe):
        pipe_element.set_seed(self.seed)
        if len(self.pipe_elements):
            pipe_element.connect_from(self.pipe_elements[-1])
        self.pipe_elements.append(pipe_element)

    async def _run_all(self):
        await asyncio.gather(*[elem.run() for elem in self.pipe_elements])

    def run(self):
        asyncio.run(self._run_all())

    def __repr__(self) -> str:
        repr = ""
        for elem in self.pipe_elements[:-1]:
            repr += str(elem) + "->"
        repr += str(self.pipe_elements[-1])
        return repr
    

    def reset(self):
        for elem in self.pipe_elements:
            elem.reset()