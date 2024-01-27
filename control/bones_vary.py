import copy
import sys
sys.path.append("..")
from simulation import Simulator, MonitorNetworkModel
import math
import numpy as np
import argparse
import os
import copy


class BONESVary(Simulator):
    def __init__(self, args):
        """
        BONES algorithm simulator with varying computation speed
        :param args: parameter settings
        """
        super(BONESVary, self).__init__(args)
        # BONES parameters
        self.gamma_p = args.gamma_p
        self.V = (args.download_buffer * 1000 - self.seg_time) * self.seg_time / (self.vmaf_down_avg[-1] + self.gamma_p)
        self.V = self.V * args.V_multiplier

        # monitor download process
        self.monitor = args.monitor
        if args.monitor:
            self.network = MonitorNetworkModel(self.logger, self.net_time, self.bandwidth, self.latency,
                                               self.monitor_hook, self.buffer_enhance, self.buffer_download)

        # automatic parameter tuning
        self.autotune = args.autotune
        if args.autotune:
            self._load_autotune_table(args)

        # varying computation speed
        self.enhance_time_vary = copy.deepcopy(self.enhance_time)
        self.noise_mean = args.noise_mean
        self.noise_range = args.noise_range

    def control_download_start(self):
        """
        Control action when download starts
        :return:
        """
        # automatic parameter tuning
        if self.autotune:
            self._tune_params()

        # "drift" part of the objective function
        buff_down = self.buffer_download.buffer_level
        buff_enh = self.buffer_enhance.buffer_level
        obj_drift = buff_down * self.seg_time + buff_enh * self.enhance_time  # (num_bitrate, num_method + 1)

        # "reward" part of the objective function
        obj_reward = self.vmaf_enh_avg + self.gamma_p  # (num_bitrate, num_method + 1)

        # objective function = (drift - V * reward) / segment size
        seg_size = self.seg_size[self.idx_down][:, None]  # (num_bitrate, 1)
        obj = (obj_drift - self.V * obj_reward) / seg_size  # (num_bitrate, num_method + 1)

        # disable enhancement option too long to finish
        mask_late = ((buff_enh + self.enhance_time - buff_down) > 0)  # (num_bitrate, num_method + 1)
        obj[mask_late] = math.inf

        # choose the action combination that minimizes the objective function
        decision = obj.argmin()
        action_down = decision // obj.shape[1]
        action_enh = decision % obj.shape[1]
        self.action_down = action_down
        self.action_enh = action_enh

        if self.logger.is_verbose():
            self.logger.write("(control) download {} enhance {}".format(
                self.action_down, self.action_enh
            ))
        return

    def _tune_params(self):
        """
        Automatic parameter tuning
        :return:
        """
        # pre-defined parameters
        num_bandwidth = 25
        min_bandwidth = 200
        step_bandwidth = 200
        num_variance = 10
        min_variance = 500
        step_variance = 500
        num_latency = 2
        min_latency = 20
        step_latency = 80

        # estimate current bandwidth, bandwidth variance, and latency
        bandwidth, latency = self.tput_history.estimate()
        variance = np.var(self.tput_history.throughput_window)

        # select optimal parameters from the pre-computed table
        def get_idx(val, min_val, step_val, num_val):
            if val < min_val:
                idx_val = 0
            else:
                idx_val = int((val - min_val) // step_val)
            idx_val = min(num_val - 1, idx_val)
            return idx_val
        idx_bandwidth = get_idx(bandwidth, min_bandwidth, step_bandwidth, num_bandwidth)
        idx_variance = get_idx(variance, min_variance, step_variance, num_variance)
        idx_latency = get_idx(latency, min_latency, step_latency, num_latency)

        # update parameters
        self.gamma_p = self.gamma_p_table[idx_bandwidth, idx_variance, idx_latency]
        self.V_multiplier = self.V_multiplier_table[idx_bandwidth, idx_variance, idx_latency]
        self.V = (self.buffer_download.max_level - self.seg_time) * self.seg_time / (self.vmaf_down_avg[-1] + self.gamma_p)
        self.V = self.V * self.V_multiplier

        if self.logger.is_verbose():
            self.logger.write("(autotune) gamma_p {} V_multiplier {}".format(
                self.gamma_p, self.V_multiplier
            ))
        return

    def control_download_finish(self):
        """
        Control action when download finishes
        :return:
        """
        # awake the enhancement buffer if it's sleeping
        if self.buffer_enhance.buffer_level == 0:
            self.session.set_enhance(False, 0)
        # push the computation task to the enhancement buffer
        if self.action_enh != 0:
            # varying computation speed
            enhance_time = self.enhance_time_vary[self.action_down][self.action_enh]
            factor = np.random.normal(self.noise_mean, self.noise_range)
            enhance_time = enhance_time * factor
            enhance_time = max(1e-6, enhance_time)
            # stall playback for computation if needed
            freeze_time = self.buffer_enhance.buffer_level + enhance_time - self.buffer_download.buffer_level
            if freeze_time > 0:
                self.buffer_download.idle_time += freeze_time
            # push the computation task
            self.buffer_enhance.push(segment_index=self.idx_enh, decision=self.action_enh,
                                 segment_time=enhance_time)
        self.idx_enh += 1

        # wait if buffer is nearly full
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
        return

    def control_enhance_finish(self):
        """
        Control action when enhancement finishes
        :return:
        """
        return

    def monitor_hook(self, buffer_download, buffer_enhance, throughput, latency, remain_size, total_time):
        """
        Hook function to monitor the download process
        :param buffer_download: download buffer level
        :param buffer_enhance: enhancement buffer leve
        :param throughput: throughput
        :param latency: latency
        :param remain_size: remaining size to download
        :param total_time: total download time
        :return: segment size of new option, or None
        """
        # compute objective score of the current option
        obj_drift = buffer_download * self.seg_time + buffer_enhance * self.enhance_time[self.action_down][self.action_enh]
        obj_reward = self.vmaf_enh_avg[self.action_down][self.action_enh] + self.gamma_p
        score_now = (obj_drift - self.V * obj_reward) / remain_size

        # compute objective score of the other options
        obj_drift = buffer_download * self.seg_time + buffer_enhance * self.enhance_time  # (num_bitrate, num_method + 1)
        obj_reward = self.vmaf_enh_avg + self.gamma_p  # (num_bitrate, num_method + 1)
        seg_size = self.seg_size[self.idx_down][:, None]  # (num_bitrate, 1)
        score_other = (obj_drift - self.V * obj_reward) / seg_size  # (num_bitrate, num_method + 1)
        mask_late = ((buffer_enhance + self.enhance_time - buffer_download) > 0)  # (num_bitrate, num_method + 1)
        score_other[mask_late] = math.inf
        decision = score_other.argmin()
        action_down = decision // score_other.shape[1]
        action_enh = decision % score_other.shape[1]

        if (action_down != self.action_down) and (action_enh != self.action_enh) and (score_other[action_down, action_enh] < score_now):
            # abort the ongoing download and switch to another option
            self.action_down = action_down
            self.action_enh = action_enh
            new_size = self.seg_size[self.idx_down][self.action_down]
            if self.logger.is_verbose():
                self.logger.write("(monitor) download {} enhance {}".format(
                    self.action_down, self.action_enh
                ))
            return new_size
        return None

    def _load_autotune_table(self, args):
        path2name = {
            "../data/bbb/nas_1080ti.json": "nas",
            "../data/bbb/imdn_div2k_2080ti.json": "div2080",
            "../data/bbb/imdn_bbb_2080ti.json": "bbb2080",
            "../data/bbb/imdn_div2k_3060ti.json": "div3060",
            "../data/bbb/imdn_bbb_3060ti.json": "bbb3060",
        }
        abs_path = os.path.dirname(os.path.abspath(__file__)) + "/"
        tables = np.load(abs_path + f"autotune_bones_{path2name[args.enhance_path]}.npz")
        self.gamma_p_table, self.V_multiplier_table = tables["gamma_p_table"], tables["V_multiplier_table"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator for neural-enhanced streaming system.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_path', default="bones_test.txt",
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

    parser.add_argument('-gp', '--gamma_p', type=float, default=10,
                        help='BONES gamma*p parameter.')
    parser.add_argument('-vm', '--V_multiplier', type=float, default=1,
                        help='BONES gamma*p parameter.')
    parser.add_argument('-m', '--monitor', action='store_true',
                        help='Monitor download process.')
    parser.add_argument('-a', '--autotune', action='store_true',
                        help='Automatic parameter tuning.')
    parser.add_argument('-nm', '--noise_mean', type=float, default=1,
                        help='Computation speed noise mean.')
    parser.add_argument('-nr', '--noise_range', type=float, default=0.1,
                        help='Computation speed noise range.')


    args = parser.parse_args()


    simulator = BONESVary(args)
    simulator.run()
    report = simulator.report()
