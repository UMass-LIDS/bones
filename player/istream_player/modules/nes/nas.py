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
from istream_player.core.bw_meter import BandwidthMeter, BandwidthUpdateListener
from istream_player.modules.nes.pensieve_model import ActorCritic
import torch
import os


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 5
VIDEO_BIT_RATES = [398, 802, 1203, 2406, 4738]  # kbits per sec
BUFFER_NORM_FACTOR = 10.0 * 1000.0  # 10 seconds, convert from milliseconds


@ModuleOption("nas", requires=[DownloadBufferImpl, EnhanceBufferImpl, Enhancer, Scheduler, BandwidthMeter], default=True)
class NASController(Module, NESController):
    log = logging.getLogger("NASController")
    def __init__(self):
        super().__init__()
        self.model = ActorCritic(False)
        ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(ABSOLUTE_PATH, "nas_actor_logb_10k.pt")
        self.model._actor.load_state_dict(torch.load(model_path))
        self.deadline_calc_method = 'min_down_seg_enh'
        self.model._actor.cpu()
        self.state = torch.zeros((1, S_INFO, S_LEN))

        self.tput_history = None
        self.last_action = 0
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

        self.tput_history = ThroughputHistory(bw_meter, 8)
        self.bw_meter.add_listener(self.tput_history)



    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Tuple[Dict[int, int], Dict[int, int]]:
        if self.seg_time is None:
            mpd = self.scheduler.mpd_provider.mpd
            self.seg_time = mpd.max_segment_duration * 1000  # ms
            self.num_seg = mpd.media_presentation_duration // mpd.max_segment_duration

        self.index = index

        download_actions = dict()
        enhance_actions = dict()

        for adaptation_set in adaptation_sets.values():
            if self.first_download:
                self.first_download = False
                download_action, enhance_action = 0, 0
            else:
                download_action, enhance_action = self.nas_decision(adaptation_set)

            download_actions[adaptation_set.id] = download_action
            enhance_actions[adaptation_set.id] = enhance_action

        return download_actions, enhance_actions

    def nas_decision(self, adaptation_set: AdaptationSet):
        bitrate = [representation.bandwidth for representation in adaptation_set.representations.values()]
        self.bitrate = np.array(bitrate)
        self.seg_size = np.array(bitrate)[:, None] * self.seg_time / 1000 # (num_bitrate, 1)

        self.buff_down = self.download_buffer.buffer_level(continuous=True) * 1000
        self.buff_enh = self.enhance_buffer.buffer_level(continuous=True) * 1000

        self.log.info(f"Download buffer level: {self.buff_down}")
        self.log.info(f"Enhance buffer level: {self.buff_enh}")

        act_nas = self._nas_control()
        action_down = act_nas
        self.last_action = act_nas

        # greedy enhancement
        action_enh = self._greedy_control(action_down)
        return action_down, action_enh

    def _nas_control(self):
        sim_state = self.observe()
        self.state = self.state.clone().detach()
        self.state = torch.roll(self.state, -1, dims=-1)

        self.state[0, 0, -1] = sim_state["last"] / float(np.max(VIDEO_BIT_RATES))  # last quality
        self.state[0, 1, -1] = sim_state["buffer"] / BUFFER_NORM_FACTOR  # 10 sec
        self.state[0, 2, -1] = float(sim_state["throughput"][-1]) / 8000  # Mega bytes per second
        self.state[0, 3, -1] = float(sim_state["delay"][-1]) / BUFFER_NORM_FACTOR  # 10 sec
        self.state[0, 4, :A_DIM] = torch.tensor(sim_state["nextSizes"])  # mega bytes
        self.state[0, 5, -1] = sim_state["remain"]

        self.action_down = self.model.actionSelect(self.state)
        return self.action_down

    def observe(self):
        """
        Observe the environment
        :return:
        """
        # length <= window size in the first few chunks
        throughput, delay = self.tput_history.get_history()
        nextChunkSizes = self.seg_size.squeeze() / 8000000.0

        state = {
            "throughput": np.array(throughput) / 1000,  # throughput of previous chunks (kbps)
            "delay": delay,  # download time of previous chunks (ms)
            "last": self.bitrate[self.last_action] / 1000,  # last chosen bitrate (kbps)
            "buffer": self.buff_down,  # current buffer level (ms)
            "nextSizes": nextChunkSizes,  # available bitrates (kbps)
            "remain": max(self.num_seg - self.index, 0) / self.num_seg,  # fraction of segments left
        }
        return state

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


class ThroughputHistory(BandwidthUpdateListener):
    def __init__(self, bw_meter, window_size):
        super().__init__()
        self.window_size = window_size
        self.bw_meter = bw_meter
        self.throughput_window = []
        self.delay_window = []

    async def on_bandwidth_update(self, bw: float) -> None:
        self.throughput_window += [bw]
        self.throughput_window = self.throughput_window[-self.window_size:]

        latest_url = max(self.bw_meter.stats.keys())
        stat = self.bw_meter.get_stats(latest_url)
        delay = (stat.stop_time - stat.start_time) * 1000.  # ms
        self.delay_window += [delay]
        self.delay_window = self.delay_window[-self.window_size:]
        return

    def get_history(self):
        return self.throughput_window, self.delay_window
