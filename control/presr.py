import sys
sys.path.append("..")
from simulation import Simulator
import math
import numpy as np
import argparse
import copy


class PreSR(Simulator):
    def __init__(self, args):
        """
        PreSR Algorithm Simulator
        :param args: parameter settings
        """
        super(PreSR, self).__init__(args)
        self.look_ahead = args.look_ahead
        self.quality_factor = args.quality_factor
        self.variation_factor = args.variation_factor
        self.rebuffer_factor = args.rebuffer_factor
        self.buffer_prefetch = []
        self.last_quality = self.vmaf_enh_seg[0, 0, 0]

        self.max_skip = 5  # max number of complex segments skipped in one turn

        # settings in the paper
        self.act_down_pre = 2  # download 480p for enhancement
        self.act_enh_pre = int(np.argmax(self.vmaf_enh_avg[self.act_down_pre]))  # choose max quality improvement
        self.threshold = 80  # quality threshold for complex segments
        self.complex = self._set_complex_tags()  # complex segment indices

    def control_download_start(self):
        """
        Control action when download starts
        :return:
        """

        # push prefetched segments to the download buffer
        finish_flag = self._push_prefetch()
        if finish_flag:
            self.idx_down = self.seg_size.shape[0] + 1
            self.idx_enh = self.seg_size.shape[0] + 1
            return

        throughput, _ = self.tput_history.estimate()
        buffer = self.buffer_download.buffer_level
        last_quality = self.last_quality
        idx_complex = self._update_complex(throughput)
        basic_set, prefetch_set = self._get_solution_set(idx_complex)

        # compute QoE scores for basic solutions
        max_score = -1E6
        max_action = 0
        if basic_set != []:
            score, action = self._solve_basic(basic_set, buffer, throughput, last_quality)
            if score > max_score:
                max_score = score
                max_action = action
        # compute QoE scores for prefetching solutions
        if prefetch_set != []:
            score, action = self._solve_prefetch(prefetch_set, idx_complex, buffer, throughput, last_quality)
            if score > max_score:
                max_score = score
                max_action = action

        if max_action == -1:
            # download complex segment
            self.action_down = self.act_down_pre
            self.action_enh = self.act_enh_pre
            self.idx_enh = idx_complex
        else:
            self.action_down = max_action
            self.action_enh = 0

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
        # awake the enhancement buffer if it's sleeping
        if self.buffer_enhance.buffer_level == 0:
            self.session.set_enhance(False, 0)

        # check whether the computation can be finished on time
        action_down = self.action_down
        action_enh = self.action_enh
        buff_down = self.buffer_download.buffer_level
        buff_enh = self.buffer_enhance.buffer_level
        if buff_down - buff_enh - self.seg_time < self.enhance_time[action_down][action_enh]:
            self.action_enh = 0

        if self.action_enh != 0:
            # push the computation task to the enhancement buffer
            self.buffer_enhance.push(segment_index=self.idx_enh, decision=self.action_enh,
                                 segment_time=self.enhance_time[self.action_down][self.action_enh])

            # pull segment out of the download buffer
            self.idx_down -= 1
            self.buffer_download.history[self.idx_down] = 0
            del self.buffer_download.content[-1]
            self.buffer_download.buffer_level -= self.seg_time

            # assume that enhancement can always be finished on time
            self.buffer_prefetch.append(self.idx_enh)

        # finish simulation
        if self.idx_down >= self.vmaf_down_seg.shape[0]:
            self.session.set_download(False, math.inf)
            self.idx_enh = self.seg_size.shape[0] + 1
            return

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
        pass
        return

    def control_enhance_finish(self):
        """
        Control action when enhancement finishes
        :return:
        """
        # Stop enhancing if no complex segment is left
        if self.complex == []:
            self.idx_enh = self.seg_size.shape[0] + 1
        return


    def _push_prefetch(self):
        """
        Push prefetched segments to the download buffer
        :return:
        """
        while (len(self.buffer_prefetch) != 0) and (self.idx_down == self.buffer_prefetch[0]):
            # the segment has been prefetched
            self.buffer_download.push(segment_index=self.idx_down, decision=self.act_down_pre,
                                     segment_time=self.seg_time)
            self.idx_down += 1
            self.buffer_prefetch.pop(0)

        if self.idx_down >= self.seg_size.shape[0]:
            # finish simulation
            self.idx_enh = self.seg_size.shape[0] + 1
            return True
        return False


    def _solve_prefetch(self, prefetch_set, idx_complex, buffer, throughput, last_quality):
        """
        Solve MPC problem for prefetching solutions
        :param prefetch_set: prefetching solution set
        :param idx_complex: index of complex segment
        :param buffer: current download buffer level
        :param throughput: estimated throughput
        :param last_quality: last segment's quality
        :return: maximum score, best action
        """
        max_score = -math.inf
        max_action = None
        pos_complex = idx_complex - self.idx_down

        for actions in prefetch_set:
            score_quality = 0
            score_variation = 0
            score_rebuffer = 0
            buf = copy.deepcopy(buffer)
            last = copy.deepcopy(last_quality)

            # generate action history
            if pos_complex < self.look_ahead:
                # complex segment inside look-ahead window
                download_history = []
                enhance_history = []
                for i in range(len(actions)):
                    if actions[i] != -1:
                        download_history.append(actions[i])
                        enhance_history.append(0)
                    if i == pos_complex:
                        download_history.append(self.act_down_pre)
                        enhance_history.append(self.act_enh_pre)
            else:
                # complex segment outside look-ahead window
                download_history = []
                enhance_history = []
                for act in actions:
                    if act != -1:
                        download_history.append(act)
                        enhance_history.append(0)

            # compute quality and quality variation score
            for idx_act in range(len(download_history)):
                act_down = download_history[idx_act]
                act_enh = enhance_history[idx_act]
                quality_now = self.vmaf_enh_seg[self.idx_down + idx_act, act_down, act_enh]
                score_quality += quality_now
                score_variation += abs(quality_now - last)
                last = quality_now
            if pos_complex >= self.look_ahead:
                # process complex segment outside look-ahead window
                score_quality += self.vmaf_enh_seg[idx_complex, self.act_down_pre, self.act_enh_pre]

            # compute rebuffer score
            for idx_act in range(len(actions)):
                act = actions[idx_act]
                if act == -1:
                    # complex segment
                    download_time = self.seg_size[idx_complex, self.act_down_pre] / throughput
                else:
                    # other segments
                    download_time = self.seg_size[self.idx_down, act] / throughput
                score_rebuffer += np.clip(download_time - buf, a_min=0, a_max=None)
                buf = np.clip(buf - download_time, a_min=0, a_max=None) + self.seg_time
                # add complex segment into buffer
                if idx_act == pos_complex:
                    buf += self.seg_time

            score_quality = self.quality_factor * score_quality / len(actions)
            if len(actions) > 1:
                score_variation = self.variation_factor * score_variation / (len(actions) - 1)
            if len(actions) > 1:
                score_rebuffer = self.rebuffer_factor * score_rebuffer / (len(actions) - 1)
            score = score_quality - score_variation - score_rebuffer

            if score > max_score:
                max_score = score
                max_action = actions[0]
        return max_score, max_action

    def _solve_basic(self, basic_set, buffer, throughput, last_quality):
        """
        Solve MPC problem for basic solution set
        :param basic_set: basic solution set
        :param buffer: current download buffer level
        :param throughput: estimated throughput
        :param last_quality: last segment's quality
        :return: maximum score, best action
        """
        max_score = -math.inf
        max_action = None

        # compute QoE scores for basic solutions
        for actions in basic_set:
            score_quality = 0
            score_variation = 0
            score_rebuffer = 0
            buf = copy.deepcopy(buffer)
            last = copy.deepcopy(last_quality)

            for idx_act in range(len(actions)):
                act = actions[idx_act]
                quality_now = self.vmaf_down_seg[self.idx_down + idx_act, act]
                score_quality += quality_now
                score_variation += abs(quality_now - last)
                download_time = self.seg_size[self.idx_down, act] / throughput
                score_rebuffer += np.clip(download_time - buf, a_min=0, a_max=None)
                buf = np.clip(buf - download_time, a_min=0, a_max=None) + self.seg_time
                last = quality_now

            score_quality = self.quality_factor * score_quality / len(actions)
            score_variation = self.variation_factor * score_variation / len(actions)
            score_rebuffer = self.rebuffer_factor * score_rebuffer / len(actions)
            score = score_quality - score_variation - score_rebuffer

            if score > max_score:
                max_score = score
                max_action = actions[0]
        return max_score, max_action

    def _update_complex(self, throughput):
        """
        Update complexity tags
        :param throughput: estimated throughput
        :return: index to the latest complex segment, or None
        """
        for i in range(self.max_skip):
            if len(self.complex) == 0:
                break
            idx_complex = self.complex.pop(0)
            playback_time = self.buffer_download.buffer_level + self.seg_time * (idx_complex - self.idx_down)
            download_time = self.seg_size[idx_complex, self.act_down_pre] / throughput
            enhance_time = download_time + np.clip(self.buffer_enhance.buffer_level - download_time, a_min=0, a_max=None) + self.enhance_time[self.act_down_pre, self.act_enh_pre]

            if playback_time >= enhance_time:
                # enhancement can be finished on time
                return idx_complex
            else:
                return None

    def _get_solution_set(self, idx_complex):
        """
        Get solution set for the next look-ahead window
        :param idx_complex: index to the next complex segment or None
        :return: basic solution set, prefetch solution set
        """
        basic_set = []
        prefetch_set = []
        # basic solution set
        num_seg = self.vmaf_enh_seg.shape[0]
        num_act = self.bitrate.shape[0]
        for act_down in range(num_act):
            base_quality = self.vmaf_down_seg[self.idx_down, act_down]
            idx_end = np.clip(self.idx_down + self.look_ahead, a_min=None, a_max=num_seg)
            actions = [act_down]
            for idx_seg in range(self.idx_down + 1, idx_end):
                # choose the quality closest to the base quality
                diff = 1E6
                act_best = -1
                for ad in range(num_act):
                    if abs(self.vmaf_down_seg[idx_seg, ad] - base_quality) < diff:
                        diff = abs(self.vmaf_down_seg[idx_seg, ad] - base_quality)
                        act_best = ad
                actions.append(act_best)
            basic_set.append(actions)

        if idx_complex is None:
            return basic_set, prefetch_set

        # pre-fetching solution set
        actions_pre = basic_set[self.act_down_pre]
        future = idx_complex - self.idx_down
        if future < self.look_ahead:
            # complex segment within the look-ahead window
            for i in range(future + 1):
                # insert pre-fetched segments
                actions = copy.deepcopy(actions_pre)
                actions = actions[:future] + actions[future + 1:]
                actions = actions[:i] + [-1] + actions[i:]
                prefetch_set.append(actions)
        else:
            # complex segment outside the look-ahead window
            for i in range(self.look_ahead):
                # insert pre-fetched segments
                actions = copy.deepcopy(actions_pre)
                actions = actions[:i] + [-1] + actions[i + 1:]
                prefetch_set.append(actions)

        return basic_set, prefetch_set


    def _set_complex_tags(self):
        """
        Set complex tags for segments
        :return:
        """
        # complexity tags
        complex = []

        # compute the number of complex segments
        num_seg = self.vmaf_enh_seg.shape[0]
        enh_speed = self.enhance_time[self.act_down_pre, self.act_enh_pre]
        num_complex = int(num_seg / (enh_speed / self.seg_time))
        num_complex = num_seg if num_complex > num_seg else num_complex

        # compute the quality gain of each segment
        quality_before = self.vmaf_down_seg[:, self.act_down_pre]
        quality_after = self.vmaf_enh_seg[:, self.act_down_pre, self.act_enh_pre]
        gain = quality_after - quality_before

        # segments passing the chunk size threshold & quality threshold & having max quality gain are complex
        avg_size = np.mean(self.seg_size[:, self.act_down_pre])
        idx_complex = np.argsort(-gain)
        gain_sort = gain[idx_complex]
        for i in range(gain_sort.shape[0]):
            if len(complex) == num_complex:
                break
            # if quality_after[i] < self.threshold:
            #     continue
            # if self.seg_size[i, self.act_down_pre] < avg_size:
            #     continue
            complex.append(idx_complex[i])
        complex.sort()
        if complex[0] == 0:
            complex.pop(0)  # the first segment cannot be complex
        return complex



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator for neural-enhanced streaming system.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_path', default="presr_test.txt",
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

    parser.add_argument('-la', '--look_ahead', default=5, type=int,
                        help='Look-ahead window size.')
    parser.add_argument('-qf', '--quality_factor', default=10, type=int,
                        help='Penalty factor for quality.')
    parser.add_argument('-vf', '--variation_factor', default=10, type=int,
                        help='Penalty factor for quality variation.')
    parser.add_argument('-rf', '--rebuffer_factor', default=1, type=int,
                        help='Penalty factor for rebuffering events.')
    args = parser.parse_args()


    simulator = PreSR(args)
    simulator.run()
    report = simulator.report()
