import abc
import asyncio
from typing import Any

import numpy as np


class SignalPipe(abc.ABC):
    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.input_queue = None
        self.output_queue = None

    def __repr__(self) -> str:
        return self.__class__.__name__

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def connect(self, input_queue: asyncio.Queue | None, output_queue: asyncio.Queue | None):
        if input_queue:
            self.input_queue = input_queue
        if output_queue:
            self.output_queue = output_queue

    def connect_to(self, other: "SignalPipe"):
        q = self.output_queue or other.input_queue or asyncio.Queue(maxsize=2)
        other.connect(q, None)
        self.connect(None, q)

    def connect_from(self, other: "SignalPipe"):
        q = other.output_queue or self.input_queue or asyncio.Queue(maxsize=2)
        other.connect(None, q)
        self.connect(q, None)

    async def run(self):
        if self.input_queue and self.output_queue:
            await self.run_pipe()
        elif self.input_queue:
            await self.run_sink()
        elif self.output_queue:
            await self.run_source()
        else:
            raise ValueError(f"Module not connected{self.__repr__()}")

    async def run_pipe(self):
        while True:
            item = await self.input_queue.get()  # pyright: ignore[reportOptionalMemberAccess]
            if item is None:
                break
            result = self.process(item)
            await self.output_queue.put(result)  # pyright: ignore[reportOptionalMemberAccess]
        await self.output_queue.put(None)  # pyright: ignore[reportOptionalMemberAccess]

    async def run_source(self):
        while True:
            item = self.generate()
            if item is None:
                break
            await self.output_queue.put(item)  # pyright: ignore[reportOptionalMemberAccess]
        await self.output_queue.put(None)  # pyright: ignore[reportOptionalMemberAccess]

    async def run_sink(self):
        while True:
            item = await self.input_queue.get()  # pyright: ignore[reportOptionalMemberAccess]
            if item is None:
                break
            self.consume(item)

    def process(self, signal: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def generate(self) -> np.ndarray | None:
        raise NotImplementedError()

    def consume(self, signal: np.ndarray):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self) -> None:
        pass


class CompoundPipe(SignalPipe):
    def __init__(self, pipes: list[SignalPipe | None], seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.pipes = [p for p in pipes if p is not None]

        for pipe in self.pipes:
            pipe.set_seed(seed)

        for i in range(len(self.pipes) - 1):
            self.pipes[i].connect_to(self.pipes[i + 1])

    def connect(self, input_queue: asyncio.Queue | None, output_queue: asyncio.Queue | None):
        super().connect(input_queue, output_queue)
        if input_queue and self.pipes:
            self.pipes[0].input_queue = input_queue
        if output_queue and self.pipes:
            self.pipes[-1].output_queue = output_queue

    async def run(self):
        await asyncio.gather(*[pipe.run() for pipe in self.pipes])

    def reset(self) -> None:
        for pipe in self.pipes:
            pipe.reset()

    def process(self, signal: np.ndarray) -> np.ndarray:
        raise NotImplementedError("CompoundPipe uses async queue-based run(), not process()")

    def generate(self) -> np.ndarray | None:
        raise NotImplementedError("CompoundPipe uses async queue-based run(), not generate()")

    def consume(self, signal: np.ndarray) -> None:
        raise NotImplementedError("CompoundPipe uses async queue-based run(), not consume()")


class Terminator(SignalPipe):
    def consume(self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]):
        return

    def reset(self) -> None:
        pass
