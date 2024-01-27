import asyncio
import logging

from istream_player.config.config import PlayerConfig
# from istream_player.core.buffer import BufferManager
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.player import Player, PlayerEventListener
from istream_player.core.scheduler import Scheduler
from istream_player.models import State
from istream_player.utils.async_utils import critical_task

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.core.enhancer import Enhancer

@ModuleOption("dash", default=True, requires=[DownloadBufferImpl, EnhanceBufferImpl, Enhancer, Scheduler, MPDProvider])
class DASHPlayer(Module, Player):
    log = logging.getLogger("DASHPlayer")

    def __init__(self):
        super().__init__()

        # State related
        self._state = State.IDLE

        # Playback related
        self._playback_started = False
        self._position = 0.0

    async def setup(self,
                    config: PlayerConfig,
                    download_buffer: DownloadBufferImpl,
                    enhance_buffer:EnhanceBufferImpl,
                    enhancer: Enhancer,
                    scheduler: Scheduler,
                    mpd_provider: MPDProvider, **kwargs
    ):
        # self.min_start_buffer_duration = config.min_start_duration
        # self.min_rebuffer_duration = config.min_rebuffer_duration
        self.time_factor = config.time_factor

        self.download_buffer = download_buffer
        self.enhance_buffer = enhance_buffer
        self.enhancer = enhancer
        self.scheduler = scheduler
        self.mpd_provider = mpd_provider

    @property
    def state(self) -> State:
        return self._state

    async def _switch_state(self, old_state: State, new_state: State):
        for listener in self.listeners:
            await listener.on_state_change(self._position, old_state, new_state)

    def stop(self) -> None:
        raise NotImplementedError

    def pause(self) -> None:
        raise NotImplementedError

    def add_listener(self, listener: PlayerEventListener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    @critical_task()
    async def run(self):
        """
        The main loop.
        This method coordinate work between different components.
        """
        await self.mpd_provider.available()
        # Start the scheduler
        self._state = State.BUFFERING
        assert self.mpd_provider.mpd is not None
        first_start_time = None
        await self._switch_state(self._state, State.BUFFERING)

        while self._state != State.END:
            index, segments = await self.download_buffer.dequeue()
            if self.state == State.BUFFERING:
                await self._switch_state(self._state, State.READY)

            self.log.info(f"Playing index: {index}")

            if first_start_time is None:
                first_start_time = min(map(lambda s: s.start_time, segments.values()))

            self._position = min(map(lambda s: s.start_time, segments.values()))

            for listener in self.listeners:
                await listener.on_position_change(self._position)
            await self._switch_state(self._state, State.READY)

            for listener in self.listeners:
                await listener.on_segment_playback_start(segments)

            duration = max(map(lambda s: s.duration, segments.values()))
            self._position += duration
            for listener in self.listeners:
                await listener.on_position_change(self._position)

            # Update for next round
            if self.download_buffer.is_empty():
                if self.scheduler.is_end:
                    await self._switch_state(self._state, State.END)
                    self._state = State.END
                    await self.download_buffer.cleanup()
                    await self.enhance_buffer.cleanup()
                    await self.enhancer.cleanup()
                    self.log.info("Player closed")
                    return
                else:
                    await self._switch_state(self._state, State.BUFFERING)
                    self._state = State.BUFFERING
            else:
                await self._switch_state(self._state, State.READY)