import asyncio
import copy
import itertools
import logging
import os.path
import tempfile
import shutil
from asyncio import Task
from typing import Dict, Optional, Set, Tuple, List

from istream_player.config.config import PlayerConfig
# from istream_player.core.abr import ABRController
from istream_player.core.nes import NESController
# from istream_player.core.buffer import BufferManager
from istream_player.core.bw_meter import BandwidthMeter
from istream_player.core.downloader import (DownloadManager, DownloadRequest,
                                            DownloadType)
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.scheduler import Scheduler, SchedulerEventListener
from istream_player.models import AdaptationSet
from istream_player.utils import critical_task

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.core.enhancer import Enhancer


@ModuleOption(
    "scheduler", default=True, requires=["segment_downloader", BandwidthMeter, DownloadBufferImpl, EnhanceBufferImpl, MPDProvider, NESController, Enhancer]
)
class SchedulerImpl(Module, Scheduler):
    log = logging.getLogger("SchedulerImpl")

    def __init__(self):
        super().__init__()

        self.adaptation_sets: Optional[Dict[int, AdaptationSet]] = None
        self.started = False

        self._task: Optional[Task] = None
        self._index = 0
        self._representation_initialized: Set[str] = set()
        self._current_download_actions: Optional[Dict[int, int]] = None
        self._current_enhance_actions: Optional[Dict[int, int]] = None

        self._end = False
        self._dropped_index = None

    async def setup(
        self,
        config: PlayerConfig,
        segment_downloader: DownloadManager,
        bandwidth_meter: BandwidthMeter,
        download_buffer: DownloadBufferImpl,
        enhance_buffer: EnhanceBufferImpl,
        mpd_provider: MPDProvider,
        nes_controller: NESController,
        enhancer: Enhancer,
    ):
        self.max_buffer_duration = config.buffer_duration
        self.update_interval = config.static.update_interval
        self.time_factor = config.time_factor
        self.run_dir = config.run_dir
        self.content_aware = config.content_aware

        self.download_manager = segment_downloader
        self.bandwidth_meter = bandwidth_meter
        self.download_buffer = download_buffer
        self.enhance_buffer = enhance_buffer
        self.nes_controller = nes_controller
        self.mpd_provider = mpd_provider
        self.enhancer = enhancer

        select_as = config.select_as.split("-")
        if len(select_as) == 1 and select_as[0].isdecimal():
            self.selected_as_start = int(select_as[0])
            self.selected_as_end = int(select_as[0])
        elif (
            len(select_as) == 2
            and (select_as[0].isdecimal() or select_as[0] == "")
            and (select_as[1].isdecimal() or select_as[1] == "")
        ):
            self.selected_as_start = int(select_as[0]) if select_as[0] != "" else None
            self.selected_as_end = int(select_as[1]) if select_as[1] != "" else None
        else:
            raise Exception("select_as should be of the format '<uint>-<uint>' or '<uint>'.")

        if self.run_dir is not None:
            if os.path.exists(self.run_dir):
                shutil.rmtree(self.run_dir)
            os.mkdir(self.run_dir)

    def segment_limits(self, adap_sets: Dict[int, AdaptationSet]) -> tuple[int, int]:
        ids = [
            [[seg_id for seg_id in repr.segments.keys()] for repr in as_val.representations.values()]
            for as_val in adap_sets.values()
        ]
        ids = itertools.chain(*ids)
        ids = list(itertools.chain(*ids))
        # print(adap_sets, ids)
        return min(ids), max(ids)

    @critical_task()
    async def run(self):
        await self.mpd_provider.available()
        assert self.mpd_provider.mpd is not None
        self.adaptation_sets = self.select_adaptation_sets(self.mpd_provider.mpd.adaptation_sets)

        # Quick start
        if not self.content_aware:
            await self.enhancer.start(self.adaptation_sets)
            self.log.info("Enhancer started")

        # Start from the min segment index
        self.first_segment, self.last_segment = self.segment_limits(self.adaptation_sets)
        self.log.info(f"{self.first_segment=}, {self.last_segment=}")
        self._index = self.first_segment
        while True:
            # Check buffer level
            if self.download_buffer.buffer_level() > self.max_buffer_duration:
                await asyncio.sleep(self.time_factor * self.update_interval)
                continue

            assert self.mpd_provider.mpd is not None

            if self.mpd_provider.mpd.type == "dynamic":
                await self.mpd_provider.update()
                self.adaptation_sets = self.select_adaptation_sets(self.mpd_provider.mpd.adaptation_sets)
                self.first_segment, self.last_segment = self.segment_limits(self.adaptation_sets)
                self.log.info(f"{self.first_segment=}, {self.last_segment=}")

            if self._index < self.first_segment:
                self.log.info(f"Segment {self._index} not in mpd, Moving to next segment")
                self._index += 1
                continue

            if self.mpd_provider.mpd.type == "dynamic" and self._index > self.last_segment:
                self.log.info(f"Waiting for more segments in mpd : {self.mpd_provider.mpd.type}")
                await asyncio.sleep(self.time_factor * self.update_interval)
                continue

            if self.content_aware and (not self.enhancer.is_ready()) and self.download_buffer.buffer_level(continuous=True) > 10:
                await self.enhancer.start(self.adaptation_sets)
                self.log.info("Enhancer started ")

            # Download one segment from each adaptation set
            if self._index == self._dropped_index:
                download_actions, enhance_actions = self.nes_controller.update_selection_lowest(self.adaptation_sets)
            else:
                download_actions, enhance_actions = self.nes_controller.update_selection(self.adaptation_sets, self._index)
            self.log.info(f"Index {self._index}, download actions {download_actions}, enhance actions {enhance_actions}")
            self._current_download_actions = download_actions
            self._current_enhance_actions = enhance_actions

            # All adaptation sets take the current bandwidth
            adap_bw = {as_id: self.bandwidth_meter.bandwidth for as_id in download_actions.keys()}

            # Get segments to download for each adaptation set
            try:
                segments = {}
                for adaptation_set_id in download_actions.keys():
                    download_action = download_actions[adaptation_set_id]
                    enhance_action = enhance_actions[adaptation_set_id]
                    adapt_set = self.adaptation_sets[adaptation_set_id]

                    # only display the video track
                    if adapt_set.content_type != "video":
                        continue

                    segment = adapt_set.representations[download_action].segments[self._index]
                    segment.download_action = download_action
                    segment.enhance_action = enhance_action
                    segments[adaptation_set_id] = segment

            except KeyError:
                # No more segments left
                self.log.info("No more segments left")
                self._end = True
                return

            for listener in self.listeners:
                await listener.on_segment_download_start(self._index, adap_bw, segments)

            # duration = 0
            urls = {}  # key: url, value: [segment, download result]
            for adaptation_set_id, download_action in download_actions.items():
                adaptation_set = self.adaptation_sets[adaptation_set_id]
                representation = adaptation_set.representations[download_action]
                representation_str = "%d-%d" % (adaptation_set_id, representation.id)
                if representation_str not in self._representation_initialized:
                    await self.download_manager.download(DownloadRequest(representation.initialization, DownloadType.STREAM_INIT))
                    init_data, _ = await self.download_manager.wait_complete(representation.initialization)
                    # save stream initialization segment locally
                    init_file = tempfile.NamedTemporaryFile(dir=self.run_dir, delete=False)
                    init_file.write(init_data)
                    init_file.close()
                    self._representation_initialized.add(representation_str)
                    for segment in representation.segments.values():
                        segment.init_path = init_file.name

                    self.log.info(f"Stream {representation_str} initialized")
                try:
                    segment = representation.segments[self._index]
                except IndexError:
                    self.log.info("Segments ended")
                    self._end = True
                    return
                urls[segment.url] = [segment, None]
                await self.download_manager.download(DownloadRequest(segment.url, DownloadType.SEGMENT))
                # duration = segment.duration
            self.log.info(f"Waiting for completion urls {urls.keys()}")

            for url in urls.keys():
                urls[url][1], _ = await self.download_manager.wait_complete(url)

            self.log.info(f"Completed downloading from urls {urls.keys()}")

            for url in urls.keys():
                segment_data = urls[url][1]
                if segment_data is None:
                    self._dropped_index = self._index
                    continue
                # save segment locally
                segment_file = tempfile.NamedTemporaryFile(dir=self.run_dir, delete=False)
                segment_file.write(segment_data)
                segment_file.close()
                urls[url][0].path = segment_file.name

            download_stats = {as_id: self.bandwidth_meter.get_stats(segment.url) for as_id, segment in segments.items()}
            for listener in self.listeners:
                await listener.on_segment_download_complete(self._index, segments, download_stats)

            await self.download_buffer.enqueue(self._index, segments)
            if self.has_enhance(self._current_enhance_actions):
                await self.enhance_buffer.enqueue(self._index, copy.deepcopy(segments))

            self._index += 1

    def select_adaptation_sets(self, adaptation_sets: Dict[int, AdaptationSet]):
        as_ids = adaptation_sets.keys()
        start = self.selected_as_start or min(as_ids)
        end = self.selected_as_end or max(as_ids)
        print(f"{start=}, {end=}")
        return {as_id: as_val for as_id, as_val in adaptation_sets.items() if as_id >= start and as_id <= end}

    async def stop(self):
        await self.download_manager.close()
        if self._task is not None:
            self._task.cancel()

    @property
    def is_end(self):
        return self._end

    def add_listener(self, listener: SchedulerEventListener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    async def cancel_task(self, index: int):
        """
        Cancel current downloading task, and move to the next one

        Parameters
        ----------
        index: int
            The index of segment to cancel
        """

        # If the index is the the index of currently downloading segment, ignore it
        if self._index != index or self._current_download_actions is None:
            return

        # Do not cancel the task for the first index
        if index == 0:
            return

        assert self.adaptation_sets is not None
        for adaptation_set_id, selection in self._current_download_actions.items():
            segment = self.adaptation_sets[adaptation_set_id].representations[selection].segments[self._index]
            self.log.debug(f"Stop current downloading URL: {segment.url}")
            await self.download_manager.stop(segment.url)

    async def drop_index(self, index):
        self._dropped_index = index

    def has_enhance(self, enhance_actions):
        for as_id in enhance_actions:
            if enhance_actions[as_id] != 0:
                return True
        return False

