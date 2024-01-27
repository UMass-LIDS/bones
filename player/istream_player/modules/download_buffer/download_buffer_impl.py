import asyncio
import logging

from typing import Dict, Tuple

from istream_player.config.config import PlayerConfig
from istream_player.core.buffer import BufferManager
from istream_player.core.module import Module, ModuleOption
from istream_player.models.mpd_objects import Segment

from queue import PriorityQueue
import math
from istream_player.core.renderer import Renderer


@ModuleOption("download_buffer", requires=[Renderer], default=True)
class DownloadBufferImpl(Module, BufferManager):
    """
    Buffer storing downloaded segments.
    Beyond queue operations, this buffer also allows replacing a downloaded segment with its enhanced counterpart.
    """
    def __init__(self) -> None:
        super().__init__()
        self.log = logging.getLogger("DownloadBuffer")

        self._buffer_level: float = 0
        self._indices = PriorityQueue()  # priority queue of segment indices
        self._segments: Dict[int, Dict[int, Segment]] = dict()  # dictionary of (segment index, segments)
        self._accessible: asyncio.Condition = asyncio.Condition()
        self._max_buffer_level = math.inf

        self._is_end = False

    async def publish_buffer_level(self):
        for listener in self.listeners:
            await listener.on_buffer_level_change(self._buffer_level)

    async def setup(self, config: PlayerConfig, renderer: Renderer, **kwargs):
        self._max_buffer_level = config.buffer_duration
        self.renderer = renderer


    async def run(self) -> None:
        await self.publish_buffer_level()

    async def enqueue(self, index:int, segments: Dict[int, Segment]) -> None:
        async with self._accessible:
            max_duration = max(map(lambda s: s[1].duration, segments.items()))

            # prevent buffer overflow
            while self._buffer_level + max_duration > self._max_buffer_level:
                await self._accessible.wait()

            self._indices.put(index)
            self._segments[index] = segments
            self._buffer_level += max_duration
            await self.publish_buffer_level()
            self._accessible.notify_all()

    async def dequeue(self) -> Tuple[int, Dict[int, Segment]]:
        async with self._accessible:
            # prevent buffer underflow
            while self._indices.empty():
                await self._accessible.wait()

                if self._is_end:
                    self.log.info("Download buffer closed.")
                    return None, None

            index = self._indices.get()
            segments = self._segments[index]
            del self._segments[index]
            max_duration = max(map(lambda s: s[1].duration, segments.items()))
            self._buffer_level -= max_duration
            await self.publish_buffer_level()
            self._accessible.notify_all()
            return index, segments

    async def replace(self, index: int, segments: Dict[int, Segment]) -> None:
        async with self._accessible:
            if index not in self._segments:
                # abort replacement
                self.log.info(f"Replacement failure. Segment {index} not found in buffer")
                return
            self._segments[index] = segments
            self._accessible.notify_all()

    def buffer_level(self, continuous: bool = False) -> float:
        if continuous:
            return self._buffer_level + self.renderer.remain_task()
        return self._buffer_level

    def is_empty(self) -> bool:
        return self._indices.empty()

    async def cleanup(self) -> None:
        async with self._accessible:
            self._is_end = True
            self._accessible.notify_all()
        return
