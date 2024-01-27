import copy


class NetworkModel():
    def __init__(self, logger, net_time, bandwidth, latency):
        """
        Network model simulating recorded network traces
        :param logger: information logger
        :param net_time: (num_period, ), duration of each network period
        :param bandwidth: (num_period, ), bandwidth of each network period
        :param latency: (num_period, ), latency of each network period
        """
        self.logger = logger
        self.net_time = net_time
        self.bandwidth = bandwidth
        self.latency = latency

        self.num_period = net_time.shape[0]  # number of network periods
        self.idx_period = -1  # index of network period
        self.time_to_next = 0  # remaining time in this network period

    def download(self, size):
        """
        Simulate the full process of downloading a segment
        :param size: size of the segment to download, measured in bits
        :return: download time, throughput, latency
        """
        assert size >= 0
        delay_lat = self.delay_latency(1)  # time to the first bit
        delay_down = self.delay_download(size)  # download time
        tput = size / delay_down  # throughput

        if self.logger.is_verbose():
            self.logger.write(
                "(download) size {:.2f} total_time {:.2f} latency {:.2f} download_time {:.2f} throughput {:.2f}"
                .format(size, delay_lat + delay_down, delay_lat, delay_down, tput
            ))
        return delay_down, tput, delay_lat

    def delay_download(self, size):
        """
        Compute delay caused by downloading task
        :param size: size of the segment to download, measured in bits
        :return: delay caused by download
        """
        total_time = 0
        while size > 0:
            current_bandwidth = self.bandwidth[self.idx_period]
            if size <= self.time_to_next * current_bandwidth:
                time = size / current_bandwidth
                total_time += time
                self.time_to_next -= time
                size = 0
            else:  # delay causes network period shift
                total_time += self.time_to_next
                size -= self.time_to_next * current_bandwidth
                self.next_period()
        return total_time

    def delay_latency(self, delay_units):
        """
        Compute delay caused by network latency
        :param delay_units: the number of "latency unit"
        :return: delay caused by latency
        """
        total_time = 0
        while delay_units > 0:
            current_latency = self.latency[self.idx_period]
            time = delay_units * current_latency
            if time <= self.time_to_next:
                total_time += time
                self.time_to_next -= time
                delay_units = 0
            else:  # delay causes network period shift
                total_time += self.time_to_next
                delay_units -= self.time_to_next / current_latency
                self.next_period()
        return total_time

    def go_by(self, time):
        """
        Time goes by
        :param time: the amount of time goes by, measured in ms
        :return:
        """
        while time > self.time_to_next:
            time -= self.time_to_next
            self.next_period()
        self.time_to_next -= time
        return

    def next_period(self):
        """
        Move to the next network period
        :return:
        """
        self.idx_period += 1
        if self.idx_period == self.num_period:
            self.idx_period = 0
        self.time_to_next = self.net_time[self.idx_period]

        if self.logger.is_verbose():
            self.logger.write("(network) period {} time {} bandwidth {} latency {}"
                              .format(self.idx_period, self.net_time[self.idx_period],
                                      self.bandwidth[self.idx_period], self.latency[self.idx_period]))
        return


class MonitorNetworkModel(NetworkModel):
    def __init__(self, logger, net_time, bandwidth, latency, monitor_hook, buffer_enhance, buffer_download):
        """
        Network model allowing to monitor the download process
        :param logger: information logger
        :param net_time: (num_period, ), duration of each network period
        :param bandwidth: (num_period, ), bandwidth of each network period
        :param latency: (num_period, ), latency of each network period
        :param monitor_hook: hook function to monitor the download process
        :param buffer_enhance: enhancement buffer
        :param buffer_download: download buffer
        """
        super(MonitorNetworkModel, self).__init__(logger, net_time, bandwidth, latency)
        self.min_monitor_size = 12000  # minimum size to download in each monitor interval
        self.min_monitor_time = 50  # minimum duration of each monitor interval
        self.monitor_hook = monitor_hook
        self.buffer_enhance = buffer_enhance
        self.buffer_download = buffer_download

    def download(self, size):
        """
        Repeatedly download small pieces and let the algorithm monitor the download process
        :param size: size of the segment to download, measured in bits
        :return: download time, throughput, latency
        """
        assert size >= 0
        # simulate the download process on an imaginary timeline
        buffer_enh = copy.deepcopy(self.buffer_enhance.buffer_level)
        buffer_down = copy.deepcopy(self.buffer_download.buffer_level)
        total_download_time = 0
        total_download_size = 0
        remain_size = copy.deepcopy(size)

        # compute latency
        latency = self.delay_latency(1)
        total_download_time += latency

        # compute download time
        while remain_size > 0:
            bits, time = self.delay_download_monitor(remain_size)
            total_download_time += time
            total_download_size += bits
            remain_size -= bits

            # simulate downloading
            buffer_enh = max(0, buffer_enh - time)
            buffer_down = max(0, buffer_down - time)

            if remain_size > 0:
                # monitor the download process
                tput = total_download_size / (total_download_time - latency)
                new_size = self.monitor_hook(buffer_down, buffer_enh, tput, latency, remain_size, total_download_time)
                if new_size is not None:
                    # abort the current download and start a new one
                    remain_size = new_size
                    total_download_time += latency  # new download incurs additional latency
                    if self.logger.is_verbose():
                        self.logger.write("(download) new size {:.2f}".format(new_size))

        # summarize the whole process as one download event
        delay_down = total_download_time - latency  # download time
        tput = total_download_size / delay_down  # throughput
        delay_lat = latency  # latency

        if self.logger.is_verbose():
            self.logger.write(
                "(download) size {:.2f} total_time {:.2f} latency {:.2f} download_time {:.2f} throughput {:.2f}"
                .format(size, delay_lat + delay_down, delay_lat, delay_down, tput
            ))
        return delay_down, tput, delay_lat

    def delay_download_monitor(self, size):
        """
        Compute download delay in a monitor interval
        :param size: size of the segment to download, measured in bits
        :return: delay caused by download
        """
        min_size = copy.deepcopy(self.min_monitor_size)
        min_time = copy.deepcopy(self.min_monitor_time)
        total_size = 0
        total_time = 0
        while size > 0 and (min_size > 0 or min_time > 0):
            current_bandwidth = self.bandwidth[self.idx_period]
            if current_bandwidth > 0:
                min_bits = max(min_size, min_time * current_bandwidth)
                bits_to_next = self.time_to_next * current_bandwidth
                # always choose the shortest time interval
                if size <= min_bits and size <= bits_to_next:
                    bits = size
                    time = bits / current_bandwidth
                    self.time_to_next -= time
                elif min_bits <= bits_to_next:
                    bits = min_bits
                    time = bits / current_bandwidth
                    min_size = 0
                    min_time = 0
                    self.time_to_next -= time
                else:
                    bits = bits_to_next
                    time = self.time_to_next
                    self.next_period()
            else:  # current_bandwidth == 0
                bits = 0
                if min_size > 0 or min_time > self.time_to_next:
                    time = self.time_to_next
                    self.next_period()
                else:
                    time = min_time
                    self.time_to_next -= time
            total_size += bits
            total_time += time
            size -= bits
            min_size -= bits
            min_time -= time
        return total_size, total_time

    def delay_latency_monitor(self, delay_units):
        """
        Compute network latency in a monitor interval
        :param delay_units: the number of "latency unit"
        :return: delay caused by latency
        """
        min_size = copy.deepcopy(self.min_monitor_size)
        min_time = copy.deepcopy(self.min_monitor_time)
        total_delay_units = 0
        total_delay_time = 0
        while delay_units > 0 and min_time > 0:
            current_latency = self.latency[self.idx_period]
            time = delay_units * current_latency
            if time <= min_time and time <= self.time_to_next:
                units = delay_units
                self.time_to_next -= time
            elif min_time <= self.time_to_next:
                # time > 0 implies current_latency > 0
                time = min_time
                units = time / current_latency
                self.time_to_next -= time
            else:
                time = self.time_to_next
                units = time / current_latency
                self.next_period()
            total_delay_units += units
            total_delay_time += time
            delay_units -= units
            min_time -= time
        return total_delay_units, total_delay_time


def test_network_model():
    import json
    import numpy as np
    from logger import Logger
    # parse network trace
    bandwidth_multiplier = 1
    trace = json.load(open("../data/trace/dummy.json"))
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
    # test
    network = NetworkModel(Logger(), net_time, bandwidth, latency)
    for i in range(100):
        size = np.random.uniform(500000, 5000000)
        time = np.random.uniform(0, 1000)
        network.download(size)
        network.go_by(time)
        print("test", i, "size", size, "time", time, "period", network.idx_period, "next", network.time_to_next)


if __name__ == '__main__':
    test_network_model()