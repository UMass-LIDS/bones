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
from istream_player.core.bw_meter import BandwidthMeter
import logging

@ModuleOption("bones", requires=[DownloadBufferImpl, EnhanceBufferImpl, Enhancer, Scheduler], default=True)
class BONESController(Module, NESController):
    log = logging.getLogger("BONESController")
    def __init__(self):
        super().__init__()
        self.gamma_p = 10.
        self.V_multiplier = 1.

        self.enhance_safety = 0.9
        self.seg_time = None

        self.first_download = True


    async def setup(self,
                    config: PlayerConfig,
                    download_buffer: DownloadBufferImpl,
                    enhance_buffer: EnhanceBufferImpl,
                    enhancer: Enhancer,
                    scheduler: Scheduler,
                    **kwargs):
        self.buffer_size = config.buffer_duration * 1000
        self.download_buffer = download_buffer
        self.enhance_buffer = enhance_buffer
        self.enhancer = enhancer
        self.scheduler = scheduler


    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Tuple[Dict[int, int], Dict[int, int]]:
        if self.seg_time is None:
            self.seg_time = self.scheduler.mpd_provider.mpd.max_segment_duration * 1000  # ms


        download_actions = dict()
        enhance_actions = dict()

        for adaptation_set in adaptation_sets.values():
            if self.first_download:
                download_action, enhance_action = 0, 0
                self.first_download = False
            else:
                download_action, enhance_action = self.bones_decision(adaptation_set)
            download_actions[adaptation_set.id] = download_action
            enhance_actions[adaptation_set.id] = enhance_action

        return download_actions, enhance_actions

    def bones_decision(self, adaptation_set: AdaptationSet):
        if not self.enhancer.is_ready():
            return 0, 0

        self.enhance_time = self.enhancer.get_latency_table() * 1000 / self.enhance_safety

        bitrate = [representation.bandwidth for representation in adaptation_set.representations.values()]
        self.bitrate = np.array(bitrate)
        self.seg_size = np.array(bitrate)[:, None] * self.seg_time / 1000 # (num_bitrate, 1)


        self.buff_down = self.download_buffer.buffer_level(continuous=True) * 1000
        self.buff_enh = self.enhance_buffer.buffer_level(continuous=True) * 1000
        self.vmaf_enh_avg = self.enhancer.get_quality_table()
        self.log.info(f"Download buffer level: {self.buff_down}")
        self.log.info(f"Enhance buffer level: {self.buff_enh}")

        self.action_down, self.action_enh = self._bones_control()

        return self.action_down, self.action_enh

    def _bones_control(self):
        V = (self.buffer_size - self.seg_time) * self.seg_time / (np.max(self.vmaf_enh_avg) + self.gamma_p)
        V = V * self.V_multiplier

        obj_drift = self.buff_down * self.seg_time + self.buff_enh * self.enhance_time  # (num_bitrate, num_method + 1)

        # "reward" part of the objective function
        obj_reward = self.vmaf_enh_avg + self.gamma_p  # (num_bitrate, num_method + 1)

        # objective function = (drift - V * reward) / segment size
        obj = (obj_drift - V * obj_reward) / self.seg_size  # (num_bitrate, num_method + 1)

        # disable enhancement option too long to finish
        mask_late = ((self.buff_enh + self.enhance_time - self.buff_down) > 0)  # (num_bitrate, num_method + 1)
        obj[mask_late] = np.inf

        # choose the action combination that minimizes the objective function
        decision = obj.argmin()
        action_down = decision // obj.shape[1]
        action_enh = decision % obj.shape[1]
        return int(action_down), int(action_enh)