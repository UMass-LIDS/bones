import sys
sys.path.append("..")
from simulation import Simulator
import math
import argparse
from control.impatient import insert_impatient_enhancement


class Buffer(Simulator):
    def __init__(self, args):
        """
        Buffer-based Algorithm Simulator
        :param args: parameter settings
        """
        super(Buffer, self).__init__(args)
        self.reservoir = args.reservoir
        self.upper_reservoir = args.upper_reservoir
        # compute bitrate change threshold
        self.threshold = self.bitrate - self.bitrate[0]
        self.threshold = (self.threshold / self.threshold[-1]) * (self.upper_reservoir - self.reservoir)
        self.threshold = (self.threshold + self.reservoir) * self.buffer_download.max_level

        # impatient enhancement, choose the maximum quality gained in real time
        if args.impatient:
            insert_impatient_enhancement(self)

    def control_download_start(self):
        """
        Control action when download starts
        :return:
        """
        buffer_level = self.buffer_download.buffer_level
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator for neural-enhanced streaming system.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_path', default="buffer_test.txt",
                        help='Path the logging file.')
    parser.add_argument('--manifest_path', default='../data/bbb/bbb_video_info.json',
                        help='Specify the .json file describing the video manifest.')
    parser.add_argument('--trace_path', default='../data/trace/sd_fs/trace0000.json',
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

    parser.add_argument('-r', '--reservoir', default=0.375,
                        help='Buffer-based algorithm bandwidth reservoir level.')
    parser.add_argument('-ur', '--upper_reservoir', default=0.9,
                        help='Buffer-based algorithm bandwidth upper reservoir level.')
    parser.add_argument('-i', '--impatient', action='store_true',
                        help='Adopt impatient enhancement.')

    args = parser.parse_args()


    simulator = Buffer(args)
    simulator.run()
    report = simulator.report()

