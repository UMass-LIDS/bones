from abc import ABC, abstractmethod
from typing import Dict

from istream_player.core.module import ModuleInterface
from istream_player.models.mpd_objects import Segment


class EnhancerEventListener(ABC):
    async def on_enhancement_start(self, segments: Dict[int, Segment]):
        """Callback executed when a segment is played by the player

        Args:
            segment (Segment): The playback segment
        """


class Enhancer(ModuleInterface, ABC):
    def __init__(self) -> None:
        self.listeners: list[EnhancerEventListener] = []

        self.quality_table = None  # enhancement quality
        self.latency_table = None  # enhancement latency

    def add_listener(self, listener: EnhancerEventListener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    @abstractmethod
    async def start(self, adaptation_sets):
        """Start the enhancer"""
        pass

    @abstractmethod
    def is_ready(self):
        return True

    @abstractmethod
    def get_latency_table(self):
        return self.latency_table

    @abstractmethod
    def get_quality_table(self):
        return self.quality_table

    def remain_task(self):
        return 0

    async def cleanup(self):
        pass