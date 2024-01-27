import copy
import math
import numpy as np

class ThroughputHistory():
    def __init__(self, logger):
        """
        Parent class of throughput history
        :param logger: information logger
        """
        self.throughput_est = None  # estimated throughput
        self.latency_est = None  # estimated latency
        self.logger = logger

    def push(self, time, tput, lat):
        """
        Push information about the lasted download into the history
        :param time: total download time
        :param tput: throughput during download
        :param lat: latency during download
        """
        raise NotImplementedError

    def estimate(self):
        """
        Estimate current throughput and latency
        :return: estimated throughput, estimated latency
        """
        if self.logger.is_verbose():
            self.logger.write("(estimate) throughput {:.2f} latency {:.2f}".format(
                self.throughput_est, self.latency_est
            ))
        return self.throughput_est, self.latency_est


class SlidingWindow(ThroughputHistory):
    def __init__(self, logger, window_size=3):
        """
        Throughput estimator based on sliding window moving average
        :param logger: information logger
        :param window_size: moving average window size
        """
        super(SlidingWindow, self).__init__(logger)
        self.window_size = window_size

        self.throughput_window = []
        self.latency_window = []

    def push(self, time, tput, lat):
        self.throughput_window += [tput]
        self.throughput_window = self.throughput_window[-self.window_size:]
        self.throughput_est = sum(self.throughput_window) / len(self.throughput_window)

        self.latency_window += [lat]
        self.latency_window = self.latency_window[-self.window_size:]
        self.latency_est = sum(self.latency_window) / len(self.latency_window)
        return


class ExponentialWindow(ThroughputHistory):
    def __init__(self, logger, segment_time, half_life=(3000, 8000), history_size=5):
        """
        Throughput estimator based on exponential window moving average
        :param logger: information logger
        :param segment_time: segment duration
        :param half_life: half life value. Dual exponential by default
        :param history_size: size of history window
        """
        super(ExponentialWindow, self).__init__(logger)
        self.half_life_t = np.array(list(half_life))  # throughput half life
        self.half_life_l = self.half_life_t / segment_time  # latency half life
        self.throughput = np.zeros(self.half_life_t.shape)
        self.latency = np.zeros(self.half_life_t.shape)
        self.weight_t = 0
        self.weight_l = 0

        # only use to record history
        self.history_size = history_size
        self.throughput_window = []
        self.latency_window = []

    def push(self, time, tput, lat):
        # record history
        self.throughput_window += [tput]
        self.throughput_window = self.throughput_window[-self.history_size:]
        self.throughput_est = sum(self.throughput_window) / len(self.throughput_window)

        self.latency_window += [lat]
        self.latency_window = self.latency_window[-self.history_size:]
        self.latency_est = sum(self.latency_window) / len(self.latency_window)

        # compute exponential moving average
        alpha_t = np.power(0.5, time / self.half_life_t)
        self.throughput = alpha_t * self.throughput + (1 - alpha_t) * tput

        alpha_l = np.power(0.5, time / self.half_life_l)
        self.latency = alpha_l * self.latency + (1 - alpha_l) * lat
        self.weight_t += time
        self.weight_l += 1

        denominator_t = 1 - np.power(0.5, self.weight_t / self.half_life_t)
        throughput_est = self.throughput / denominator_t
        self.throughput_est = np.min(throughput_est)  # conservative min

        denominator_l = 1 - np.power(0.5, self.weight_l / self.half_life_l)
        latency_est = self.latency / denominator_l
        self.latency_est = np.max(latency_est)  # conservative max
        return


class HarmonicMean(ThroughputHistory):
    def __init__(self, logger, window_size=5):
        """
        Throughput estimator based on harmonic mean
        :param logger: information logger
        :param window_size: history window size
        """
        super(HarmonicMean, self).__init__(logger)
        self.window_size = window_size

        self.throughput_window = []
        self.latency_window = []

    def push(self, time, tput, lat):
        self.throughput_window += [tput]
        self.throughput_window = self.throughput_window[-self.window_size:]
        self.throughput_est = self.compute_harmonic_mean(self.throughput_window)

        self.latency_window += [lat]
        self.latency_window = self.latency_window[-self.window_size:]
        self.latency_est = self.compute_harmonic_mean(self.latency_window)
        return

    @staticmethod
    def compute_harmonic_mean(history):
        result = 0
        for ele in history:
            result += 1. / ele
        result = 1. / (result / len(history))
        return result


def test_throughput_estimator():
    import json
    import matplotlib.pyplot as plt
    from simulation.logger import Logger

    segment_time = 3000

    # parse network trace
    def parse_trace(trace_path, bandwidth_multiplier=1):
        trace = json.load(open(trace_path))
        net_time = []
        bandwidth = []
        latency = []
        for i in range(len(trace)):
            net_time.append(trace[i]['duration_ms'])
            bandwidth.append(trace[i]['bandwidth_kbps'])
            latency.append(trace[i]['latency_ms'])
        net_time = np.array(net_time)
        bandwidth = np.array(bandwidth) * bandwidth_multiplier
        latency = np.array(latency)
        return net_time, bandwidth, latency

    net_time, bandwidth, latency = parse_trace("network.json")

    # run estimators
    def run_estimator(estimator):
        np.random.seed(0)
        timeline_est = [0]
        bandwidth_est = [0]
        latency_est = [0]
        for i in range(net_time.shape[0]):
            duration = net_time[i]
            # split = np.random.uniform(0, duration, num_split-1)
            # split.sort()
            # split = np.concatenate([np.zeros((1, )), split, np.array([duration])], axis=0)
            num_split = duration // segment_time
            for j in range(num_split):
                # time = split[j+1] - split[j]
                time = duration / num_split
                tput = bandwidth[i]
                lat = latency[i]
                estimator.push(time, tput, lat)
                timeline_est.append(timeline_est[-1] + time)
                tput_est, lat_est = estimator.estimate()
                bandwidth_est.append(tput_est)
                latency_est.append(lat_est)
        return timeline_est, bandwidth_est, latency_est

    estimator = SlidingWindow(Logger())
    timeline_slid, bandwidth_slid, latency_slid = run_estimator(estimator)
    estimator = ExponentialWindow(Logger(), segment_time=segment_time)
    timeline_ewma, bandwidth_ewma, latency_ewma = run_estimator(estimator)
    estimator = HarmonicMean(Logger())
    timeline_harm, bandwidth_harm, latency_harm = run_estimator(estimator)

    # real network trace
    timeline_real = np.zeros(2 * net_time.shape[0])
    timeline_real[1] = net_time[0]
    for i in range(net_time.shape[0] - 1):
        timeline_real[2 * i + 1] = timeline_real[2 * i] + net_time[i]
        timeline_real[2 * i + 2] = timeline_real[2 * i] + net_time[i]
    timeline_real[-1] = timeline_real[-2] + net_time[-1]

    bandwidth_real = np.zeros(timeline_real.shape)
    latency_real = np.zeros(timeline_real.shape)
    for i in range(bandwidth_real.shape[0]):
        bandwidth_real[i] = bandwidth[i // 2]
        latency_real[i] = latency[i // 2]

    plt.plot(timeline_real, bandwidth_real, label="real")
    plt.plot(timeline_slid, bandwidth_slid, label="slid")
    plt.plot(timeline_ewma, bandwidth_ewma, label="ewma")
    plt.plot(timeline_harm, bandwidth_harm, label="harm")
    plt.xlabel("time (ms)")
    plt.ylabel("bandwidth (kbps)")
    plt.legend()
    plt.show()

    plt.plot(timeline_real, latency_real, label='real')
    plt.plot(timeline_slid, latency_slid, label='slid')
    plt.plot(timeline_ewma, latency_ewma, label='ewma')
    plt.plot(timeline_harm, latency_harm, label="harm")
    plt.xlabel("time (ms)")
    plt.ylabel("latency (ms)")
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    test_throughput_estimator()