import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from istream_player.core.module import ModuleInterface
from istream_player.models.mpd_objects import Segment


class BufferEventListener(ABC):
    async def on_buffer_level_change(self, buffer_level: float):
        pass


class BufferManager(ModuleInterface, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.listeners: list[BufferEventListener] = []

    def add_listener(self, listener: BufferEventListener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    @abstractmethod
    def buffer_level(self) -> float:
        """
        Returns
        -------
        buffer_level: float
            Current buffer level in seconds
        """
        pass

    @abstractmethod
    async def enqueue(self, index: int, segments: Dict[int, Segment]) -> None:
        """
        Enqueue some buffers into the buffer manager

        Parameters
        ----------
        index: int
            The index of the segment
        segments: Dict[int, Segment]
            The map of adaptation_id to downloaded segment
        """
        pass

    @abstractmethod
    async def dequeue(self) -> Tuple[Dict[int, Segment], float]:
        """Remove last segment from buffer"""
        pass


    @abstractmethod
    def is_empty(self) -> bool:
        """
        Returns if True there are no segments in buffer
        """
        pass
