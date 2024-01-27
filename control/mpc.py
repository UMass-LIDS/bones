import copy
import os.path
import sys
sys.path.append("..")
from simulation import Simulator
import math
import argparse
import numpy as np
from control.impatient import insert_impatient_enhancement

class MPC(Simulator):
    def __init__(self, args):
        """
        Fast MPC Algorithm Simulator
        :param args: parameter settings
        """
        super(MPC, self).__init__(args)
        self.look_ahead = args.look_ahead
        self.quality_factor = args.quality_factor
        self.variation_factor = args.variation_factor
        self.rebuffer_factor = args.rebuffer_factor
        abs_path = os.path.dirname(os.path.abspath(__file__)) + "/"
        self.table_path = abs_path + "mpc_table_q{}v{}r{}.npy".format(self.quality_factor, self.variation_factor, self.rebuffer_factor)
        self.last_action = 0
        self.util_down_avg = np.log(self.bitrate / self.bitrate[0])

        # hyper-parameters
        self.num_bin_tput = 100  # number of throughput bins
        self.min_tput = 0.1
        self.max_tput = 30000
        self.size_bin_tput = (self.max_tput - self.min_tput) / self.num_bin_tput
        self.num_bin_buffer = 100  # number of buffer level bins
        self.min_buffer = 0
        self.max_buffer = self.buffer_download.max_level
        self.size_bin_buffer = (self.max_buffer - self.min_buffer) / self.num_bin_buffer

        # generate decision table if not exist
        self.table = None
        if not os.path.exists(self.table_path):
            # generate all possible actions for reuse
            self.action_list = [[]]
            while True:
                act = self.action_list[0]
                if len(act) < self.look_ahead:
                    self.action_list.pop(0)
                    for j in range(self.bitrate.shape[0]):
                        self.action_list.append(act + [j])
                else:
                    break
            self.table = self._generate_table()

        else:
            self.table = np.load(self.table_path)

        # impatient enhancement, choose the maximum quality gained in real time
        if args.impatient:
            insert_impatient_enhancement(self)

    def _solve_one(self, buffer, throughput, last_action):
        """
        Solve MPC problem for one step
        :param buffer: current buffer level
        :param throughput: estimated throughput
        :param last_action: last action
        :return: next action
        """
        max_score = -math.inf
        max_action = None
        # brute force solution
        for actions in self.action_list:
            score_quality = 0
            score_variation = None
            score_rebuffer = 0
            buf = buffer  # future buffer level
            last = last_action

            for act in actions:
                score_quality += self.util_down_avg[act]
                if score_variation is None:
                    score_variation = 0
                else:
                    score_variation += abs(self.util_down_avg[act] - self.util_down_avg[last])
                download_time = self.seg_time * self.bitrate[act] / throughput
                tmp = download_time - buf
                score_rebuffer = score_rebuffer + tmp if tmp > 0 else score_rebuffer
                tmp = buf - download_time
                buf = tmp + self.seg_time if tmp > 0 else self.seg_time
                last = act

            # ignore startup delay
            score_quality = self.quality_factor * score_quality / len(actions)
            if len(actions) > 1:
                score_variation = self.variation_factor * score_variation / (len(actions) - 1)
            score_rebuffer = self.rebuffer_factor * score_rebuffer / len(actions)
            score = score_quality - score_variation - score_rebuffer

            if score > max_score:
                max_score = score
                max_action = actions[0]
        return max_action

    def _generate_table(self):
        """
        Generate MPC decision table
        :return:
        """
        print("Generate MPC decision table ...")
        table = np.zeros((self.num_bin_buffer, self.num_bin_tput, self.bitrate.shape[0]), dtype=int)
        for i in range(self.num_bin_buffer):
            print("{}/{}".format(i, self.num_bin_buffer))
            for j in range(self.num_bin_tput):
                for k in range(self.bitrate.shape[0]):
                    buffer = self.min_buffer + i * self.size_bin_buffer
                    throughput = self.min_tput + j * self.size_bin_tput
                    last_action = k
                    action = self._solve_one(buffer, throughput, last_action)
                    table[i, j, k] = action
        np.save(self.table_path, table)
        return table

    def control_download_start(self):
        """
        Control action when download starts
        :return:
        """
        buffer = self.buffer_download.buffer_level
        buffer = self.max_buffer if buffer > self.max_buffer else buffer
        buffer = int((buffer - self.min_buffer) // self.size_bin_buffer)
        throughput, _ = self.tput_history.estimate()
        throughput = self.max_tput if throughput > self.max_tput else throughput
        throughput = int((throughput - self.min_tput) // self.size_bin_tput)

        self.action_down = self.table[buffer, throughput, self.last_action]
        self.last_action = self.action_down

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
    parser.add_argument('--log_path', default="mpc_test.txt",
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
                        help='Throughput estimator.')
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

    parser.add_argument('-la', '--look_ahead', default=5, type=int,
                        help='Look-ahead window size.')
    parser.add_argument('-qf', '--quality_factor', default=10, type=int,
                        help='Penalty factor for quality.')
    parser.add_argument('-vf', '--variation_factor', default=10, type=int,
                        help='Penalty factor for quality variation.')
    parser.add_argument('-rf', '--rebuffer_factor', default=1, type=int,
                        help='Penalty factor for rebuffering events.')
    parser.add_argument('-i', '--impatient', action='store_true',
                        help='Adopt impatient enhancement.')
    args = parser.parse_args()


    simulator = MPC(args)
    simulator.run()
    report = simulator.report()