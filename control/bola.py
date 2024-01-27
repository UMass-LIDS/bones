import sys
sys.path.append("..")
from simulation import Simulator, MonitorNetworkModel
import argparse
from control.impatient import insert_impatient_enhancement
import numpy as np

class BOLA(Simulator):
    def __init__(self, args):
        """
        BOLA Algorithm Simulator
        :param args: parameter settings
        """
        super(BOLA, self).__init__(args)
        # BOLA parameters
        self.util_down_avg = np.log(self.bitrate / self.bitrate[0])
        self.gamma_p = args.gamma_p
        self.Vp = (args.download_buffer * 1000 - self.seg_time) / (self.util_down_avg[-1] + self.gamma_p)
        self.last_action = 0


        # monitor download process
        if args.monitor:
            self.network = MonitorNetworkModel(self.logger, self.net_time, self.bandwidth, self.latency,
                                               self.monitor_hook, self.buffer_enhance, self.buffer_download)

        # impatient enhancement, choose the maximum quality gained in real time
        if args.impatient:
            insert_impatient_enhancement(self)

    def control_download_start(self):
        """
        Control action when download starts
        :return:
        """
        best_action = self._bola_control()
        self.action_down = best_action
        self.last_action = best_action
        if self.logger.is_verbose():
            self.logger.write("(control) download {} enhance {}".format(
                self.action_down, self.action_enh
            ))
        return

    def _bola_control(self):
        buffer = self.buffer_download.buffer_level
        num_action_down = self.bitrate.shape[0]
        best_score = None
        best_action = None

        # dynamic V value
        # t = min(self.idx_down, self.seg_size.shape[0] - self.idx_down)
        # t = max(t / 2, 3)
        # buffer_dynamic = min(buffer, t * self.seg_time)
        # self.Vp = (buffer_dynamic - self.seg_time) / (self.quality_down_avg[-1] + self.gamma_p)

        # choose the bitrate that maximize "drift plus penalty"
        for action_down in range(num_action_down):
            score = (self.Vp * (self.util_down_avg[action_down] + self.gamma_p) - buffer) / self.bitrate[action_down]
            if best_score is None or score > best_score:
                best_action = action_down
                best_score = score

        # BOLA-U trick to reduce oscillation
        if best_action > self.last_action:
            action_tput = self._throughput_control()
            if best_action <= action_tput:
                pass
            elif self.last_action > action_tput:
                best_action = self.last_action
            else:
                best_action = self.last_action + 1
        return best_action

    def _throughput_control(self):
        """
        Choose the highest bitrate under the estimated throughput
        :return:
        """
        tput_est, lat_est = self.tput_history.estimate()
        act = 0
        while ((act + 1) < self.bitrate.shape[0]) and \
                lat_est + (self.seg_time * self.bitrate[act + 1]) / tput_est <= self.seg_time:
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
        self.session.set_enhance(False, np.inf)
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
        # compute the score of ongoing download option
        score_now = (self.Vp * (self.util_down_avg[self.action_down] + self.gamma_p) - buffer_download) / remain_size
        new_act = None
        # compute the score of other download options
        for act in range(self.bitrate.shape[0]):
            if act == self.action_down:
                continue
            other_size = self.seg_size[self.idx_down][act]
            score_other = (self.Vp * (self.util_down_avg[act] + self.gamma_p) - buffer_download) / other_size
            if other_size < remain_size and score_other > score_now:
                # abort the ongoing download and switch to another option
                new_act = act

        if new_act is not None:
            self.action_down = new_act
            self.last_action = new_act
            new_size = self.seg_size[self.idx_down][self.action_down]
            if self.logger.is_verbose():
                self.logger.write("(monitor) download {} enhance {}".format(
                    self.action_down, self.action_enh
                ))
            return new_size
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator for neural-enhanced streaming system.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_path', default="bola_test.txt",
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

    parser.add_argument('-gp', '--gamma_p', type=float, default=5,
                        help='BOLA gamma_p parameter, measured in seconds.')
    parser.add_argument('-m', '--monitor', action='store_true',
                        help='Monitor download process.')
    parser.add_argument('-i', '--impatient', action='store_true',
                        help='Adopt impatient enhancement.')

    args = parser.parse_args()


    simulator = BOLA(args)
    simulator.run()
    simulator.report()
