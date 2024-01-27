import sys
sys.path.append("..")
import json
import numpy as np
import argparse
import math
from .tput_history import SlidingWindow, ExponentialWindow, HarmonicMean
from .network_model import NetworkModel
from .buffer_model import BufferModel
from .session_model import SessionModel
from .logger import Logger


class Simulator():
    def __init__(self, args):
        """
        Parent class of simulator for neural-enhanced streaming system
        :param args: parameter settings
        """
        # initialize variables
        self.args = args  # parameter settings
        self.seg_time = None  # segment duration, (1, )
        self.bitrate = None  # bitrate ladder, (num_bitrate, )
        self.seg_size = None  # segment size, (num_segment, num_bitrate)
        self.net_time = None  # network period duration, (num_period, )
        self.bandwidth = None  # period bandwidth, (num_period, )
        self.latency = None  # period latency, (num_period, )
        self.enhance_time = None  # enhancement task duration, (num_bitrate, num_method + 1)
        self.vmaf_down_seg = None  # download quality for each segment, VMAF, (num_segment, num_bitrate)
        self.vmaf_enh_seg = None  # enhanced quality for each segment, VMAF, (num_segment, num_bitrate, num_method + 1)
        self.vmaf_down_avg = None  # average download quality, VMAF, (num_bitrate, )
        self.vmaf_enh_avg = None  # average enhanced quality, VMAF, (num_bitrate, num_method + 1)
        self.ssim_down_seg = None  # download quality for each segment, SSIM, (num_segment, num_bitrate)
        self.ssim_enh_seg = None  # enhanced quality for each segment, SSIM, (num_segment, num_bitrate, num_method + 1)
        self.global_time = 0  # global timestamp
        self.idx_down = 0  # index to the video segment waiting for download decision
        self.idx_enh = 0  # index to the video segment waiting for enhancement decision
        self.last_download = None  # last download information
        self.action_down = math.inf  # next download action, invalid by default
        self.action_enh = 0  # next enhance action, no enhance (0) by default
        self.extra_download = 0  # extra download size

        # parse input files
        self._parse_manifest(args.manifest_path, args.video_length)
        self._parse_trace(args.trace_path, args.bandwidth_multiplier)
        if args.enhance_path == '':
            # estimate quality using bitrate if not provided
            self._estimate_quality()
        else:
            self._parse_enhance(args.enhance_path, args.speed_multiplier)
        # logger
        self.logger = Logger(args.log_path, args.verbose, self.get_timestamp)

        # throughput estimator
        if args.throughput == "ewma":
            self.tput_history = ExponentialWindow(self.logger, self.seg_time)
        elif args.throughput == "sw":
            self.tput_history = SlidingWindow(self.logger)
        elif args.throughput == "hm":
            self.tput_history = HarmonicMean(self.logger)
        else:
            raise NotImplementedError

        # network model
        self.network = NetworkModel(self.logger, self.net_time, self.bandwidth, self.latency)

        # buffer model
        self.buffer_download = BufferModel(self.logger, args.download_buffer * 1000, self.seg_size.shape[0], "download")
        self.buffer_enhance = BufferModel(self.logger, args.enhance_buffer * 1000, self.seg_size.shape[0], "enhance")

        # session model
        self.session = SessionModel(self.logger)

    def run(self):
        """
        Main function that performs simulation
        There are two timelines for download buffer and enhance buffer respectively
        In each timeline, alternatively enter processing session and waiting session
        :return:
        """
        # warm-up stage
        if self.logger.is_verbose():
            self.logger.write("(warmup_stage)")
        self.warmup_stage()

        # main stage
        if self.logger.is_verbose():
            self.logger.write("(main_stage)")
        while self.is_running():
            # get index of next session, remaining time of current session
            idx_session, time_session = self.session.next()

            # time goes by
            self.global_time += time_session
            self.buffer_download.consume(time_session)
            self.buffer_enhance.consume(time_session)
            if not self.session.is_downloading:
                # not downloading
                self.network.go_by(time_session)

            # decision session
            if idx_session == self.session.DOWNLOAD_START:
                # finish waiting, start downloading
                self.control_download_start()
                if not self.is_running():
                    break
                # new download task
                size = self.seg_size[self.idx_down, self.action_down] + self.extra_download
                self.extra_download = 0
                self.last_download = self.network.download(size)
                # new download session
                cd_download = self.last_download[0] + self.last_download[2]  # total download time
                self.session.set_download(True, cd_download)

            elif idx_session == self.session.DOWNLOAD_FINISH:
                # finish downloading, start waiting
                self.tput_history.push(*self.last_download)
                assert not math.isinf(self.action_down)
                self.buffer_download.push(segment_index=self.idx_down, decision=self.action_down, segment_time=self.seg_time)

                self.idx_down += 1
                # check buffer level to avoid overflow
                self.session.set_download(False, 0)  # immediately download next segment
                self.control_download_finish()
                if not self.is_running():
                    break

            elif idx_session == self.session.ENHANCE_START:
                # finish waiting, start enhancing
                self.control_enhance_start()
                if not self.is_running():
                    break
                # new enhance session
                cd_enhance = self.buffer_enhance.get_rest()
                if cd_enhance is not None:
                    self.session.set_enhance(True, cd_enhance)
                else:
                    # no segment in buffer, sleep forever until awoken
                    self.session.set_enhance(False, math.inf)

            else:  # idx_session == self.session.ENHANCE_FINISH
                # finish enhancing, start waiting
                self.session.set_enhance(False, 0)  # immediately enhance next segment
                self.control_enhance_finish()
                if not self.is_running():
                    break

            self.validate()

        # final stage
        self.final_stage()
        return

    def report(self, title=""):
        """
        Report performance records
        :param title: title of the report
        :return: a dict of records
        """
        history_down = self.buffer_download.history
        history_enh = self.buffer_enhance.history

        # review decision history
        num_seg = self.seg_size.shape[0]
        quality_vmaf = np.zeros((num_seg, ))
        oscillation_vmaf = 0
        quality_ssim = np.zeros((num_seg, ))
        oscillation_ssim = 0
        total_bandwidth = np.zeros((1, ), dtype=np.int64)

        for i in range(num_seg):
            action_down = history_down[i]
            action_enh = history_enh[i]
            quality_vmaf[i] = self.vmaf_enh_seg[i][action_down][action_enh]
            quality_ssim[i] = self.ssim_enh_seg[i][action_down][action_enh]
            if i > 0:
                oscillation_vmaf += abs(quality_vmaf[i] - quality_vmaf[i - 1])
                oscillation_ssim += abs(quality_ssim[i] - quality_ssim[i - 1])
            total_bandwidth += self.seg_size[i][action_down]

        quality_vmaf = np.mean(quality_vmaf)
        quality_ssim = np.mean(quality_ssim)
        oscillation_vmaf = oscillation_vmaf / (num_seg - 1)
        oscillation_ssim = oscillation_ssim / (num_seg - 1)
        avg_freeze = self.buffer_download.idle_time / self.seg_size.shape[0]
        avg_bandwidth = total_bandwidth / num_seg / self.seg_time

        qoe = quality_vmaf - oscillation_vmaf - avg_freeze / 10

        report_simulator = {
            "playback time": self.global_time,
            "average quality vmaf": quality_vmaf,
            "average oscillation vmaf": oscillation_vmaf,
            "average quality ssim": quality_ssim,
            "average oscillation ssim": oscillation_ssim,
            "freeze time": self.buffer_download.idle_time,
            "freeze event": self.buffer_download.idle_event,
            "average bandwidth": avg_bandwidth,
            "qoe": qoe,
            "download history": history_down,
            "enhance history": history_enh
        }
        if self.logger.is_verbose():
            title = "Simulator Report" if title == "" else title
            self.logger.summarize(title, report_simulator)
        return report_simulator

    def control_enhance_finish(self):
        """
        Control action when enhancement finishes
        :return:
        """
        raise NotImplementedError

    def control_enhance_start(self):
        """
        Control action when enhancement starts
        :return:
        """
        raise NotImplementedError

    def control_download_finish(self):
        """
        Control action when download finishes
        :return:
        """
        raise NotImplementedError

    def control_download_start(self):
        """
        Control action when download starts
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        Validate simulator states
        :return: True if running, False otherwise
        """
        if self.buffer_download.buffer_level > self.buffer_download.max_level + 1E-6:
            error_msg = f" Enhancement: {self.args.enhance_path}, Trace: {self.args.trace_path}, Download Buffer: {self.buffer_download.buffer_level}, Enhancement Buffer: {self.buffer_enhance.buffer_level}"
            raise RuntimeError("Download buffer overflow." + error_msg)
        if self.buffer_enhance.buffer_level > self.buffer_enhance.max_level + 1E-6:
            error_msg = f" Enhancement: {self.args.enhance_path}, Trace: {self.args.trace_path}, Download Buffer: {self.buffer_download.buffer_level}, Enhancement Buffer: {self.buffer_enhance.buffer_level}"
            raise RuntimeError("Enhance buffer overflow." + error_msg)
        if self.buffer_enhance.buffer_level > self.buffer_download.buffer_level + 1E-6:
            error_msg = f" Enhancement: {self.args.enhance_path}, Trace: {self.args.trace_path}, Download Buffer: {self.buffer_download.buffer_level}, Enhancement Buffer: {self.buffer_enhance.buffer_level}"
            raise RuntimeError("Late computation." + error_msg)
        return True

    def final_stage(self):
        """
        Play out rest content in the download buffer
        :return:
        """
        if self.logger.is_verbose():
            self.logger.write("(final_stage)")

        # clear both buffers, assume enhance buffer level < download buffer level
        final_time = self.buffer_download.buffer_level
        self.global_time += final_time
        self.buffer_download.consume(final_time)
        self.buffer_enhance.consume(final_time)
        if self.logger.is_verbose():
            self.logger.write("(clear) buffer {:.2f}".format(final_time))
        return

    def warmup_stage(self):
        """
        Warm up control algorithm and probe network condition
        Download the first segment in the lowest quality without enhancement
        :return:
        """
        if self.logger.is_verbose():
            self.logger.write("(control) download 0 enhance 0")

        self.action_down = 0
        self.action_enh = 0
        size = self.seg_size[0][0]
        delay_down, tput, delay_lat = self.network.download(size)
        self.tput_history.push(delay_down, tput, delay_lat)
        total_time = delay_down + delay_lat
        self.global_time += total_time
        self.buffer_download.consume(total_time)
        self.buffer_enhance.consume(total_time)
        self.buffer_download.push(0, 0, self.seg_time)
        # exclude startup latency from rebuffering time
        self.buffer_download.idle_time = 0
        self.buffer_enhance.idle_event = 0
        self.idx_down = 1
        self.idx_enh = 1
        return

    def _estimate_quality(self):
        """
        Estimate download quality using logarithm of bitrate
        :return:
        """
        quality_down_avg = np.log(self.bitrate)
        self.vmaf_down_avg = quality_down_avg - quality_down_avg[0]
        quality_down_seg = np.log(self.seg_size / self.seg_time)
        self.vmaf_down_seg = quality_down_seg - quality_down_avg[0]
        # set enhancement consumption and quality improvement to 0
        self.enhance_time = np.zeros((1, ))
        self.vmaf_enh_avg = self.vmaf_down_avg[:, None]
        self.vmaf_enh_seg = self.vmaf_down_seg[:, None]
        return

    def _parse_enhance(self, enhance_path, speed_multiplier=1):
        """
        Parse enhancement performance file
        :param enhance_path: path to the enhancement performance file
        :param speed_multiplier: multiplier for enhancement speed
        :return:
        """
        data = self.load_json(enhance_path)
        frame_rate = data['frame_rate']  # (1, )
        enhance_speed = np.array(data['enhancement_fps'])  # (num_bitrate - 1, num_method)
        num_bitrate = enhance_speed.shape[0] + 1
        num_method = enhance_speed.shape[1]

        def parse_quality(metric):
            if metric == "ssim":
                max_score = 1
            else:  # vmaf
                max_score = 100
            quality_down_seg = np.array(data[f'base_quality_{metric}'])  # (num_bitrate - 1, num_segment)
            quality_enh_seg = np.array(data[f'enhanced_quality_{metric}'])  # (num_bitrate_enh, num_method, num_segment)
            num_bitrate_enh = quality_enh_seg.shape[0]

            num_segment = quality_down_seg.shape[1]

            # the highest quality has max score
            quality_down_seg = quality_down_seg.transpose()  # (num_segment, num_bitrate - 1)
            quality_down_seg = np.concatenate([quality_down_seg, np.full((num_segment, 1), max_score)],
                                              axis=1)  # (num_segment, num_bitrate)
            # forbid enhancing the highest quality by setting quality to -inf
            quality_enh_seg = quality_enh_seg.transpose((2, 0, 1))  # (num_segment, num_bitrate_enh, num_method)
            quality_enh_seg[quality_enh_seg < 0] = -math.inf  # invalid enhancement
            quality_enh_seg = np.concatenate([quality_enh_seg, np.full((num_segment, num_bitrate - num_bitrate_enh, num_method), -math.inf)],
                                             axis=1)  # (num_segment, num_bitrate, num_method)
            quality_enh_seg = np.concatenate([quality_down_seg[:, :, None], quality_enh_seg],
                                             axis=2)  # (num_segment, num_bitrate, num_method + 1)
            return quality_down_seg, quality_enh_seg

        # index 0 for no-enhance option
        # forbid enhancing the highest quality by setting computation time to inf
        enhance_time = frame_rate * self.seg_time / enhance_speed / speed_multiplier  # (num_bitrate - 1, num_method)
        enhance_time[enhance_time < 0] = 0  # invalid enhancement
        enhance_time = np.concatenate([enhance_time, np.full((1, num_method), 0)],
                                      axis=0)  # (num_bitrate, num_method)
        enhance_time = np.concatenate([np.full((num_bitrate, 1), 0), enhance_time],
                                      axis=1)  # (num_bitrate, num_method + 1)


        # quality metrics
        self.ssim_down_seg, self.ssim_enh_seg = parse_quality('ssim')
        self.vmaf_down_seg, self.vmaf_enh_seg = parse_quality('vmaf')
        self.vmaf_down_avg = np.mean(self.vmaf_down_seg, axis=0)
        self.vmaf_enh_avg = np.mean(self.vmaf_enh_seg, axis=0)

        self.enhance_time = enhance_time
        return

    def _parse_trace(self, trace_path, bandwidth_multiplier=1):
        """
        Parse network trace file
        :param trace_path: path to the network trace file
        :param bandwidth_multiplier: multiplier for bandwidth
        :return: network period duration (num_period, ), period bandwidth (num_period, ), period latency (num_period, )
        """
        data = self.load_json(trace_path)
        net_time = []
        bandwidth = []
        latency = []
        for i in range(len(data)):
            net_time.append(data[i]['duration_ms'])
            bandwidth.append(data[i]['bandwidth_kbps'])
            latency.append(data[i]['latency_ms'])
        net_time = np.array(net_time)
        bandwidth = np.array(bandwidth) * bandwidth_multiplier
        latency = np.array(latency)

        self.net_time = net_time
        self.bandwidth = bandwidth
        self.latency = latency
        return

    def _parse_manifest(self, manifest_path, video_length):
        """
        Parse video manifest file
        :param manifest_path: path to the video manifest file
        :param video_length: video length measured in seconds
        :return:
        """
        manifest = self.load_json(manifest_path)
        seg_time = manifest['segment_duration_ms']
        bitrate = manifest['bitrates_kbps']
        bitrate = np.array(bitrate)
        seg_size = manifest['segment_sizes_bits']
        # truncate / extend video if needed
        assert video_length is None
        # todo: truncate / extend enhance data
        # if video_length is not None:
        #     l1 = len(seg_size)
        #     l2 = math.ceil(args.video_length * 1000 / seg_time)
        #     seg_size *= math.ceil(l2 / l1)
        #     seg_size = seg_size[0:l2]
        seg_size = np.array(seg_size)

        self.seg_time = seg_time
        self.bitrate = bitrate
        self.seg_size = seg_size
        return

    def is_running(self):
        """
        Check if the simulator should be running
        :return:
        """
        num_seg = self.seg_size.shape[0]
        if (self.idx_down < num_seg) and (self.idx_enh < num_seg):
            return True
        return False

    @staticmethod
    def load_json(path):
        with open(path) as file:
            obj = json.load(file)
        return obj

    def get_timestamp(self):
        timestamp = ""
        timestamp += "[t" + str(round(self.global_time)) + "-"
        timestamp += "d" + str(self.idx_down) + "-"
        timestamp += "e" + str(self.idx_enh) + "] "
        return timestamp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator for neural-enhanced streaming system.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_path', default=None,
                        help='Path the logging file.')
    parser.add_argument('--manifest_path', default='movie.json',
                        help='Specify the .json file describing the video manifest.')
    parser.add_argument('--trace_path', default='network.json',
                        help='Specify the .json file describing the network trace.')
    parser.add_argument('--enhance_path', default='',
                        help='Specify the .json file describing the enhancement performance.')
    parser.add_argument('--video_length', default=None,
                        help='Customized video length, measured in seconds.')
    parser.add_argument('--throughput', default='ewma',
                        help='Throughput estimator. Dual exponential window moving average by default.')
    parser.add_argument('--download_buffer', default=25,
                        help='Maximum download buffer size, measured in seconds.')
    parser.add_argument('--enhance_buffer', default=25,
                        help='Maximum enhance buffer size, measured in seconds.')
    parser.add_argument('--bandwidth_multiplier', default=1, type=float,
                        help='Multiplier for the network bandwidth.')
    parser.add_argument('--speed_multiplier', default=1, type=float,
                        help='Multiplier for the enhancement speed.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose information.')
    args = parser.parse_args()

    simulator = Simulator(args)