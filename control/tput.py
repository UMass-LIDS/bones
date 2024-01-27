import sys
sys.path.append("..")
from simulation import Simulator, MonitorNetworkModel
import math
import argparse
from control.impatient import insert_impatient_enhancement


class Throughput(Simulator):
    def __init__(self, args):
        """
        Throughput-based Algorithm Simulator
        """
        super(Throughput, self).__init__(args)
        self.safety = args.safety

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
        # compute safe bandwidth
        tput_est, lat_est = self.tput_history.estimate()
        tput_safe = tput_est * self.safety

        # choose max bitrate under safe bandwidth
        act = 0
        while ((act + 1) < self.bitrate.shape[0]) and \
                lat_est + (self.seg_time * self.bitrate[act + 1]) / tput_safe <= self.seg_time:
            act += 1

        self.action_down = act
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
        abandon_multiplier = 1.8
        abandon_grace_time = 500

        if total_time > abandon_grace_time:
            estimate_time_left = remain_size / throughput
            if total_time + estimate_time_left > abandon_multiplier * self.seg_time:
                # choose new bitrate
                new_act = 0
                tput_safe = throughput * self.safety
                while ((new_act + 1) < self.bitrate.shape[0]) and \
                        latency + (self.seg_time * self.bitrate[new_act + 1]) / tput_safe <= self.seg_time:
                    new_act += 1

                new_size = self.seg_size[self.idx_down][new_act]
                if new_act < self.action_down and new_size < remain_size:
                    self.action_down = new_act
                    if self.logger.is_verbose():
                        self.logger.write("(monitor) download {} enhance {}".format(
                            self.action_down, self.action_enh
                        ))
                    return new_size
        return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator for neural-enhanced streaming system.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_path', default="tput_test.txt",
                        help='Path the logging file.')
    parser.add_argument('--manifest_path', default='../data/bbb/bbb_video_info.json',
                        help='Specify the .json file describing the video manifest.')
    parser.add_argument('--trace_path', default='../data/trace/sd_fs/trace0000.json',
                        help='Specify the .json file describing the network trace.')
    parser.add_argument('--enhance_path', default='../data/bbb/nas_1080ti.json',
                        help='Specify the .json file describing the enhancement performance.')
    parser.add_argument('--video_length', default=None,
                        help='Customized video length, measured in seconds.')
    parser.add_argument('-tp', '--throughput', default='ewma',
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

    parser.add_argument('-s', '--safety', default=0.9, type=float,
                        help='Throughput-based algorithm bandwidth safety factor.')
    parser.add_argument('-m', '--monitor', action='store_true',
                        help='Monitor download process.')
    parser.add_argument('-i', '--impatient', action='store_true',
                        help='Adopt impatient enhancement.')

    args = parser.parse_args()


    simulator = Throughput(args)
    simulator.run()
    report = simulator.report()
