from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
from istream_player.config.config import PlayerConfig
from istream_player.core.nes import NESController
# from istream_player.core.buffer import BufferManager
from istream_player.core.module import Module, ModuleOption
from istream_player.models import AdaptationSet

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.core.enhancer import Enhancer
from istream_player.core.scheduler import Scheduler
import logging
from istream_player.core.bw_meter import BandwidthMeter

@ModuleOption("buffer_greedy", requires=[DownloadBufferImpl, EnhanceBufferImpl, Enhancer, Scheduler, BandwidthMeter], default=True)
class BufferGreedyController(Module, NESController):
    log = logging.getLogger("BufferGreedyController")
    def __init__(self):
        super().__init__()
        # BOLA parameters

        self.reservoir = 0.375
        self.upper_reservoir = 0.9

        self.enhance_safety = 0.9

        self.seg_time = None

        self.first_download = True


    async def setup(self,
                    config: PlayerConfig,
                    download_buffer: DownloadBufferImpl,
                    enhance_buffer: EnhanceBufferImpl,
                    enhancer: Enhancer,
                    scheduler: Scheduler,
                    bw_meter: BandwidthMeter,
                    **kwargs):
        self.buffer_size = config.buffer_duration * 1000
        self.download_buffer = download_buffer
        self.enhance_buffer = enhance_buffer
        self.enhancer = enhancer
        self.scheduler = scheduler
        self.bw_meter = bw_meter



    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Tuple[Dict[int, int], Dict[int, int]]:
        if self.seg_time is None:
            self.seg_time = self.scheduler.mpd_provider.mpd.max_segment_duration * 1000  # ms


        download_actions = dict()
        enhance_actions = dict()

        for adaptation_set in adaptation_sets.values():
            if self.first_download:
                self.first_download = False
                download_action, enhance_action = 0, 0
            else:
                download_action, enhance_action = self.buffer_greedy_decision(adaptation_set)

            download_actions[adaptation_set.id] = download_action
            enhance_actions[adaptation_set.id] = enhance_action

        return download_actions, enhance_actions

    def buffer_greedy_decision(self, adaptation_set: AdaptationSet):
        bitrate = [representation.bandwidth for representation in adaptation_set.representations.values()]
        self.bitrate = np.array(bitrate)
        self.util_down_avg = np.log(self.bitrate / np.min(self.bitrate))

        self.buff_down = self.download_buffer.buffer_level(continuous=True) * 1000
        self.buff_enh = self.enhance_buffer.buffer_level(continuous=True) * 1000

        self.threshold = self.bitrate - self.bitrate[0]
        self.threshold = (self.threshold / self.threshold[-1]) * (self.upper_reservoir - self.reservoir)
        self.threshold = (self.threshold + self.reservoir) * self.buffer_size

        self.log.info(f"Download buffer level: {self.buff_down}")
        self.log.info(f"Enhance buffer level: {self.buff_enh}")

        action_down = self._buffer_control()

        # greedy enhancement
        action_enh = self._greedy_control(action_down)
        return action_down, action_enh

    def _greedy_control(self, action_down):
        self.enhance_time = self.enhancer.get_latency_table()
        self.vmaf_enh_avg = self.enhancer.get_quality_table()
        if self.enhance_time is None or self.vmaf_enh_avg is None:
            return 0

        self.enhance_time = self.enhance_time * 1000 / self.enhance_safety

        best_quality = 0
        best_action = 0
        for idx_action in range(self.enhance_time.shape[1]):
            if self.enhance_time[action_down][idx_action] < self.seg_time:
                if self.vmaf_enh_avg[action_down][idx_action] > best_quality:
                    best_quality = self.vmaf_enh_avg[action_down][idx_action]
                    best_action = idx_action

        action_enh = best_action
        if self.buff_down - self.buff_enh - self.seg_time <= self.enhance_time[action_down][action_enh]:
            self.action_enh = 0
        return self.action_enh

    def _buffer_control(self):
        """
        BOLA algorithm
        :return:
        """
        buffer_level = self.buff_down
        num_action_down = self.bitrate.shape[0]

        # piecewise linear mapping from buffer level to bitrate
        self.action_down = 0
        if buffer_level > self.threshold[-1]:
            # upper reservoir zone
            self.action_down = num_action_down - 1
        else:
            # reservoir & cushion zone
            for action_download in range(num_action_down):
                if buffer_level < self.threshold[action_download]:
                    self.action_down = action_download
                    break
        return self.action_down
