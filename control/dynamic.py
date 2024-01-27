import sys
sys.path.append("..")
from simulation import Simulator, MonitorNetworkModel
import math
import argparse
from control import BOLA, Throughput
from control.impatient import insert_impatient_enhancement
import numpy as np

class Dynamic(Simulator):
    def __init__(self, args):
        """
        Dynamic Algorithm Simulator
        :param args: parameter settings
        """
        super(Dynamic, self).__init__(args)
        # Dynamic parameters
        self.switch = args.switch  # switch threshold
        self.use_tput = True

        # BOLA parameters
        self.util_down_avg = np.log(self.bitrate / self.bitrate[0])
        self.gamma_p = args.gamma_p
        self.Vp = (args.download_buffer * 1000 - self.seg_time) / (self.util_down_avg[-1] + self.gamma_p)
        self.last_action = 0

        # Throughput parameters
        self.safety = args.safety

        # monitor download process
        if args.monitor:
            self.network = MonitorNetworkModel(self.logger, self.net_time, self.bandwidth, self.latency,
                                               Throughput.monitor_hook.__get__(self), self.buffer_enhance, self.buffer_download)

        # impatient enhancement, choose the maximum quality gained in real time
        if args.impatient:
            insert_impatient_enhancement(self)

    def control_download_start(self):
        """
        Control action when download starts
        :return:
        """
        act_tput = self._tput_control()
        act_bola = self._bola_control()
        buff_down = self.buffer_download.buffer_level
        quality_tput = self.util_down_avg[act_tput]
        quality_bones = self.util_down_avg[act_bola]

        # switch between throughput-based and BOLA algorithm
        if self.use_tput and buff_down >= self.switch and quality_bones >= quality_tput:
            self.use_tput = False
            self.network.monitor_hook = BOLA.monitor_hook.__get__(self)
        if (not self.use_tput) and buff_down < self.switch and quality_bones < quality_tput:
            self.use_tput = True
            self.network.monitor_hook = Throughput.monitor_hook.__get__(self)

        if self.use_tput:
            self.action_down = act_tput
            self.last_action = act_tput
            self.action_enh = 0
        else:
            self.action_down = act_bola
            self.last_action = act_bola
            self.action_enh = 0

        if self.logger.is_verbose():
            self.logger.write("(control) download {} enhance {}".format(
                self.action_down, self.action_enh
            ))
        return

    def _bola_control(self):
        """
        BOLA algorithm
        :return:
        """
        buffer = self.buffer_download.buffer_level
        num_action_down = self.bitrate.shape[0]
        best_score = None
        best_action = None

        # choose the bitrate that maximize "drift plus penalty"
        for action_down in range(num_action_down):
            score = (self.Vp * (self.util_down_avg[action_down] + self.gamma_p) - buffer) / self.bitrate[action_down]
            if best_score is None or score > best_score:
                best_action = action_down
                best_score = score

        # BOLA-U trick to reduce oscillation
        if best_action > self.last_action:
            action_tput = self._tput_control()
            if best_action <= action_tput:
                pass
            elif self.last_action > action_tput:
                best_action = self.last_action
            else:
                best_action = self.last_action + 1
        return best_action

    def _tput_control(self):
        """
        Throughput-based algorithm
        :return:
        """
        tput_est, lat_est = self.tput_history.estimate()
        tput_safe = tput_est * self.safety
        act = 0
        while ((act + 1) < self.bitrate.shape[0]) and \
                lat_est + (self.seg_time * self.bitrate[act + 1]) / tput_safe <= self.seg_time:
            act += 1
        return act

    def control_download_finish(self):
        """
        Control action when download finishes
        :return:
        """
        buffer_level = self.buffer_download.buffer_level
        max_level = self.buffer_download.max_level
        time_wait = buffer_level + self.seg_time - max_level
        time_wait = max(0, time_wait)
        self.session.set_download(False, time_wait)
        return

    def control_enhance_start(self):
        """
        Control action when enhancement starts
        :return:
        """
        # no enhancement
        self.session.set_enhance(False, math.inf)
        return

    def control_enhance_finish(self):
        """
        Control action when enhancement finishes
        :return:
        """
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator for neural-enhanced streaming system.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_path', default="dynamic_test.txt",
                        help='Path the logging file.')
    parser.add_argument('--manifest_path', default='../data/bbb/bbb_video_info.json',
                        help='Specify the .json file describing the video manifest.')
    parser.add_argument('--trace_path', default='../data/trace/dummy.json',
                        help='Specify the .json file describing the network trace.')
    parser.add_argument('--enhance_path', default='../data/bbb/nas_1080ti.json',
                        help='Specify the .json file describing the enhancement performance.')
    parser.add_argument('--video_length', default=None,
                        help='Customized video length, measured in seconds.')
    parser.add_argument('--throughput', default='ewma',
                        help='Throughput estimator. Dual exponential window moving average by default.')
    parser.add_argument('-db', '--download_buffer', default=60, type=int,
                        help='Maximum download buffer size, measured in seconds.')
    parser.add_argument('-eb', '--enhance_buffer', default=60, type=int,
                        help='Maximum enhance buffer size, measured in seconds.')
    parser.add_argument('-bm', '--bandwidth_multiplier', default=1, type=float,
                        help='Multiplier for the network bandwidth.')
    parser.add_argument('-sm', '--speed_multiplier', default=1, type=float,
                        help='Multiplier for the enhancement speed.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose information.')

    parser.add_argument('-sw', '--switch', default=10000, type=float,
                        help='Buffer level threshold for algorithm swtiching.')
    parser.add_argument('-gp', '--gamma_p', type=float, default=5,
                        help='BOLA gamma_p parameter, measured in seconds.')
    parser.add_argument('-s', '--safety', default=0.9, type=float,
                        help='Throughput-based algorithm bandwidth safety factor.')
    parser.add_argument('-m', '--monitor', action='store_true',
                        help='Monitor download process.')
    parser.add_argument('-i', '--impatient', action='store_true',
                        help='Adopt impatient enhancement.')
    args = parser.parse_args()


    simulator = Dynamic(args)
    simulator.run()
    simulator.report()

