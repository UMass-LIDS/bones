import json
import sys

sys.path.append("..")
from simulation import Simulator, ThroughputHistory
import math
import argparse
from rl.pensieve import actor_critic_model as pensieve
import numpy as np
import torch
from os import listdir, mkdir
from os.path import isfile, join, exists
from os.path import basename
from control.impatient import insert_impatient_enhancement

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 5
VIDEO_BIT_RATES = [398, 802, 1203, 2406, 4738]  # kbits per sec
BUFFER_NORM_FACTOR = 10.0 * 1000.0  # 10 seconds, convert from milliseconds


class Pensieve(Simulator):
    def __init__(self, args):
        """
        Pensieve Algorithm Simulator
        Download: Download chunks to maximize a cost function which rewards higher bit-rates, and penalizes rebuffering and bit-rate oscillation (lack of smoothness)
        Enhance: No enhancement
        :param args: parameter settings
        """
        super(Pensieve, self).__init__(args)
        self.tput_history = PensieveThroughput(self.logger, window_size=8)
        self.model = pensieve.ActorCritic(False)
        self.model._actor.load_state_dict(torch.load(args.model_path))
        self.model._actor.cpu()
        self.state = torch.zeros((1, S_INFO, S_LEN))

        # fix random seed
        np.random.seed(0)
        torch.manual_seed(0)

        # impatient enhancement, choose the maximum quality gained in real time
        if args.impatient:
            insert_impatient_enhancement(self)

    def observe(self):
        """
        Observe the environment
        :return:
        """
        # length <= window size in the first few chunks
        throughput, delay = self.tput_history.get_history()
        nextChunkSizes = np.zeros(self.seg_size.shape[1])
        if self.idx_down < self.seg_size.shape[0]:
            nextChunkSizes = self.seg_size[self.idx_down] / 8000000.0

        state = {
            "throughput": throughput,  # throughput of previous chunks (kbps)
            "delay": delay,  # download time of previous chunks (ms)
            "last": self.bitrate[self.action_down],  # last chosen bitrate (kbps)
            "buffer": self.buffer_download.buffer_level,  # current buffer level (ms)
            "nextSizes": nextChunkSizes,  # available bitrates (kbps)
            "remain": (self.seg_size.shape[0] - self.idx_down) / (self.seg_size.shape[0]),  # fraction of segments left
        }
        return state

    def control_download_start(self):
        """
        Control action when download starts
        :return:
        """
        # choose max bitrate under safe bandwidth
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

        if self.logger.is_verbose():
            self.logger.write("(control) download {} enhance {}".format(
                self.action_down, self.action_enh
            ))
        return

    def control_download_finish(self):
        """
        Control action when download finishes
        :return:
        """
        # check buffer level to avoid overflow
        buffer_level = self.buffer_download.buffer_level
        max_level = self.buffer_download.max_level
        time_wait = buffer_level + self.seg_time - max_level
        # wait if buffer is nearly full
        time_wait = max(0, time_wait)
        self.session.set_download(False, time_wait)
        return

    def control_enhance_start(self):
        """
        Control action when enhancement starts
        :return:
        """
        # disable enhancement
        self.session.set_enhance(False, math.inf)
        return

    def control_enhance_finish(self):
        """
        Control action when enhancement finishes
        :return:
        """
        # disable enhancement
        self.session.set_enhance(False, math.inf)
        return


class PensieveThroughput(ThroughputHistory):
    def __init__(self, logger, window_size):
        """
        Throughput history for Pensieve
        :param logger: information logger
        :param window_size: history window size
        """
        super(PensieveThroughput, self).__init__(logger)
        self.window_size = window_size
        self.throughput_window = []
        self.delay_window = []

    def push(self, time, tput, lat):
        self.throughput_window += [tput]
        self.throughput_window = self.throughput_window[-self.window_size:]

        self.delay_window += [time + lat]  # total download time = download time + latency
        self.delay_window = self.delay_window[-self.window_size:]
        return

    def get_history(self):
        """
        Get the history of throughput and delay
        :return: throughput history, delay history
        """
        return self.throughput_window, self.delay_window


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator for neural-enhanced streaming system.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_path', default="pensieve.txt",
                        help='Path the logging file.')
    parser.add_argument('--manifest_path', default='../data/bbb/bbb_video_info.json',
                        help='Specify the .json file describing the video manifest.')
    parser.add_argument('--model_path', default='../rl/pensieve/actor_logb_10k.pt',
                        help='Specify the .json file describing the video manifest.')
    parser.add_argument('--trace_path', default='../data/trace/dummy.json',
                        help='Specify the .json file describing the network trace.')
    parser.add_argument('--enhance_path', default='../data/bbb/nas_1080ti.json',
                        help='Specify the .json file describing the enhancement performance.')
    parser.add_argument('--video_length', default=None,
                        help='Customized video length, measured in seconds.')
    parser.add_argument('--throughput', default='ewma',
                        help='Throughput estimator. Dual exponential window moving average by default.')
    parser.add_argument('--bandwidth_multiplier', default=1, type=float,
                        help='Multiplier for the network bandwidth.')
    parser.add_argument('--speed_multiplier', default=1, type=float,
                        help='Multiplier for the enhancement speed.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose information.')
    parser.add_argument('--download_buffer', default=60,
                        help='Maximum download buffer size, measured in seconds.')
    parser.add_argument('--enhance_buffer', default=60,
                        help='Maximum enhance buffer size, measured in seconds.')
    parser.add_argument('-i', '--impatient', action='store_true',
                        help='Adopt impatient enhancement.')
    args = parser.parse_args()

    simulator = Pensieve(args)
    simulator.run()
    report = simulator.report()