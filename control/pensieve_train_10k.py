import sys
sys.path.append("..")
from simulation import Simulator, ThroughputHistory
import numpy as np
import argparse
import logging
import argparse
import numpy as np
import torch.multiprocessing as mp
import torch
import pensieve as pensieve_test
from rl.pensieve import actor_critic_model as pensieve
from datetime import datetime
from os import listdir, mkdir, system
from os.path import isfile, join, exists
from tqdm import tqdm


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 5
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 4
TRAIN_SEQ_LEN = 159  # take as a train batch
MODEL_SAVE_INTERVAL = 100
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
# REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
REBUF_PENALTY = 2.66
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = '../rl/pensieve/results_10k/'
LOG_FILE = '../rl/pensieve/results_10k/log'
TEST_LOG_FOLDER = '../rl/pensieve/test_results/'
TRACES_PATH = '../data/trace/fcc_10k/'
VIDEO_BIT_RATES = [398, 802, 1203, 2406, 4738] # kbits per sec
BUFFER_NORM_FACTOR = 10.0*1000.0 # 10 seconds, convert from milliseconds
TRACES = [join(TRACES_PATH, f) for f in listdir(TRACES_PATH) if isfile(join(TRACES_PATH, f))]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CRITIC_MODEL= './results/critic.pt'
# ACTOR_MODEL = './results/actor.pt'
CRITIC_MODEL = None

TOTALEPOCH = 60000
IS_CENTRAL = True
NO_CENTRAL = False

def testing(epoch, actor_model, log_file):
    return
    # clean up the test results folder
    system('rm -r ' + TEST_LOG_FOLDER)
    system('mkdir ' + TEST_LOG_FOLDER)
    # run test script
    report = pensieve_test.main()
    log_file.write(str(epoch) + '\t' +
                   str(report)+ '\t')
    log_file.flush()


def central_agent(net_params_queues, exp_queues):
    torch.set_num_threads(1)

    timenow = datetime.now()

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    net = pensieve.ActorCritic(True)
    test_log_file = open(LOG_FILE + '_test', 'w')

    for epoch in tqdm(range(TOTALEPOCH - 1)):
        # synchronize the network parameters of work agent
        actor_net_params = net.getActorParam()
        for i in range(NUM_AGENTS):
            net_params_queues[i].put(actor_net_params)

        # record average reward and td loss change
        # in the experiences from the agents
        total_batch_len = 0.0
        total_reward = 0.0
        total_td_loss = 0.0
        total_entropy = 0.0
        total_agents = 0.0

        # assemble experiences from the agents
        for i in range(NUM_AGENTS):
            s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

            net.getNetworkGradient(s_batch, a_batch, r_batch, terminal=terminal)

            total_reward += np.sum(r_batch)
            total_batch_len += len(r_batch)
            total_agents += 1.0
            total_entropy += np.sum(info['entropy'])

        # log training information
        net.updateNetwork()

        avg_reward = total_reward / total_agents
        avg_entropy = total_entropy / total_batch_len

        logging.info('Epoch: ' + str(epoch) +
                     ' Avg_reward: ' + str(avg_reward) +
                     ' Avg_entropy: ' + str(avg_entropy))
        # print(epoch)

        if (epoch + 1) % MODEL_SAVE_INTERVAL == 0:
            # Save the neural net parameters to disk.
            # print("\nTrain ep:" + str(epoch + 1) + ",time use :" + str((datetime.now() - timenow).seconds) + "s\n")
            timenow = datetime.now()
            torch.save(net._actor.state_dict(), SUMMARY_DIR + "/actor.pt")
            torch.save(net._critic.state_dict(), SUMMARY_DIR + "/critic.pt")
            testing(epoch + 1, SUMMARY_DIR + "/actor.pt", test_log_file)

def get_next_trace(epoch_num):
    next_trace_num = epoch_num % len(TRACES)
    trace = TRACES[next_trace_num]
    # print(trace)
    return trace


def agent(agent_id, net_params_queue, exp_queue, pensieveArgs):
    torch.set_num_threads(1)
    with open(LOG_FILE + '_agent_' + str(agent_id), 'w') as log_file:
        net = pensieve.ActorCritic(False)
        # initial synchronization of the network parameters from the coordinator
        time_stamp = 0

        for epoch in range(TOTALEPOCH):
            trace = get_next_trace(epoch)
            pensieveArgs.trace_path = trace
            simulator = PensieveTrain(pensieveArgs)
            actor_net_params = net_params_queue.get()
            net.hardUpdateActorNetwork(actor_net_params)
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY
            s_batch = []
            a_batch = []
            r_batch = []
            entropy_record = []
            state = torch.zeros((1, S_INFO, S_LEN))


            sim_state = simulator.observe()
            end_of_video = sim_state["remain"] == 0
            last_util = -1
            while not end_of_video and len(s_batch) < TRAIN_SEQ_LEN:
                last_bit_rate = bit_rate

                state = state.clone().detach()

                state = torch.roll(state, -1, dims=-1)

                state[0, 0, -1] = sim_state["last"] / float(np.max(VIDEO_BIT_RATES))  # last quality
                state[0, 1, -1] = sim_state["buffer"] / BUFFER_NORM_FACTOR  # 10 sec
                state[0, 2, -1] = float(sim_state["throughput"][-1]) / 8000  # Mega bytes per second
                state[0, 3, -1] = float(sim_state["delay"][-1]) / BUFFER_NORM_FACTOR  # 10 sec
                state[0, 4, :A_DIM] = torch.tensor(sim_state["nextSizes"])   # mega bytes
                state[0, 5, -1] = sim_state["remain"]

                bit_rate = net.actionSelect(state.to(DEVICE))

                sim_reward = simulator.act(bit_rate)
                sim_state = simulator.observe()
                end_of_video = sim_state["remain"] == 0

                util = sim_reward["util"]
                oscillation = abs(util - last_util) if last_util > 0 else 0
                # reward = util - SMOOTH_PENALTY * oscillation - REBUF_PENALTY * sim_reward["rebuffer"] / 1000.
                reward = util - oscillation - 0.1 * sim_reward["rebuffer"]
                last_util = util

                s_batch.append(state)
                a_batch.append(bit_rate)
                r_batch.append(reward)
                entropy_record.append(3)

                # log time_stamp, bit_rate, buffer_size, reward
                log_file.write(str(time_stamp) + '\t' +
                               str(VIDEO_BIT_RATES[bit_rate]) + '\t' +
                               str(sim_state["buffer"]) + '\t' +
                               str(sim_reward["rebuffer"]) + '\t' +
                               str(sim_state["last"]) + '\t' +
                               str(sim_state["delay"]) + '\t' +
                               str(reward) + '\n')
                log_file.flush()

            exp_queue.put([s_batch,
                           a_batch,
                           r_batch,
                           end_of_video,
                           {'entropy': entropy_record}])

            log_file.write('\n')


def main():
    time = datetime.now()
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    torch.multiprocessing.set_start_method('spawn')

    # create result directory
    if not exists(SUMMARY_DIR):
        mkdir(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, net_params_queues[i], exp_queues[i], args)))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()
    for i in range(NUM_AGENTS):
        agents[i].join()

    print(str(datetime.now() - time))


class PensieveTrain(Simulator):
    def __init__(self, args):
        """
        Simulator only used for training Pensieve
        :param args: parameter settings
        """
        super(PensieveTrain, self).__init__(args)
        self.tput_history = PensieveThroughput(self.logger, window_size=8)
        # self.util_down_seg = np.log(self.bitrate / self.bitrate[0])  # utility of downloading each segment
        self.util_down_avg = self.vmaf_down_avg
        # download the first chunk in the lowest quality for warm-up
        self.warmup_stage()
        self.last_rebuffer = self.buffer_download.idle_time  # previous rebuffering time

    def observe(self):
        """
        Observe the environment
        :return:
        """
        # length <= window size in the first few chunks
        throughput, delay = self.tput_history.get_history()
        nextChunkSizes = np.zeros(self.seg_size.shape[1])
        if self.idx_down < self.seg_size.shape[0]:
            nextChunkSizes = self.seg_size[self.idx_down] / 8000000.0

        state = {
            "throughput": throughput,  # throughput of previous chunks (kbps)
            "delay": delay,  # download time of previous chunks (ms)
            "last": self.bitrate[self.action_down],  # last chosen bitrate (kbps)
            "buffer": self.buffer_download.buffer_level,  # current buffer level (ms)
            "nextSizes": nextChunkSizes,  # available bitrates (kbps)
            "remain": (self.seg_size.shape[0] - self.idx_down)/(self.seg_size.shape[0]),  # fraction of segments left
        }
        return state


    def act(self, idx_action):
        """
        Take action
        :param idx_action: index of the action
        :return:
        """
        self.action_down = idx_action

        bitrate = self.bitrate[self.action_down]
        util = self.util_down_avg[self.action_down]
        rebuffer = self._act()
        reward = {
            "bitrate": bitrate,  # bitrate of this segment, can be used to estimate utility
            "util": util,  # utility of this segment
            "rebuffer": rebuffer,  # rebuffering time for this segment, measured in ms
        }
        return reward


    def _act(self):
        # start download
        size = self.seg_size[self.idx_down, self.action_down]
        self.last_download = self.network.download(size)
        time_download = self.last_download[0] + self.last_download[2]  # total download time = download time + latency

        # downloading
        self.buffer_download.consume(time_download)

        # finish download
        self.tput_history.push(*self.last_download)
        self.buffer_download.push(segment_index=self.idx_down, decision=self.action_down, segment_time=self.seg_time)
        self.idx_down += 1

        # wait to avoid buffer overflow
        buffer_level = self.buffer_download.buffer_level
        max_level = self.buffer_download.max_level
        time_wait = buffer_level + self.seg_time - max_level
        time_wait = max(0, time_wait)
        self.buffer_download.consume(time_wait)

        # compute rebuffering time
        rebuffer = self.buffer_download.idle_time - self.last_rebuffer
        self.last_rebuffer = self.buffer_download.idle_time
        return rebuffer


class PensieveThroughput(ThroughputHistory):
    def __init__(self, logger, window_size):
        """
        Throughput history for Pensieve
        :param logger: information logger
        :param window_size: history window size
        """
        super(PensieveThroughput, self).__init__(logger)
        self.window_size = window_size
        self.throughput_window = []
        self.delay_window = []

    def push(self, time, tput, lat):
        self.throughput_window += [tput]
        self.throughput_window = self.throughput_window[-self.window_size:]

        self.delay_window += [time + lat]  # total download time = download time + latency
        self.delay_window = self.delay_window[-self.window_size:]
        return

    def get_history(self):
        """
        Get the history of throughput and delay
        :return: throughput history, delay history
        """
        return self.throughput_window, self.delay_window


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator for neural-enhanced streaming system.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_path', default="../rl/pensieve/pensieve_log.txt",
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
    parser.add_argument('--download_buffer', default=60,
                        help='Maximum download buffer size, measured in seconds.')
    parser.add_argument('--enhance_buffer', default=60,
                        help='Maximum enhance buffer size, measured in seconds.')
    parser.add_argument('--bandwidth_multiplier', default=1, type=float,
                        help='Multiplier for the network bandwidth.')
    parser.add_argument('--speed_multiplier', default=1, type=float,
                        help='Multiplier for the enhancement speed.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose information.')
    args = parser.parse_args()

    main()
