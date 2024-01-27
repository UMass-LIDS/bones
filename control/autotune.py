import sys
sys.path.append("..")
import json
import numpy as np
import matplotlib.pyplot as plt
# import bocd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from control import BONES, BOLA
import argparse
import copy
import os

target_name = "nas"

name2path = {
    "nas": "../data/bbb/nas_1080ti.json",
    "div2080": "../data/bbb/imdn_div2k_2080ti.json",
    "bbb2080": "../data/bbb/imdn_bbb_2080ti.json",
    "div3060": "../data/bbb/imdn_div2k_3060ti.json",
    "bbb3060": "../data/bbb/imdn_bbb_3060ti.json",
}

path2name = {
    "../data/bbb/nas_1080ti.json": "nas",
    "../data/bbb/imdn_div2k_2080ti.json": "div2080",
    "../data/bbb/imdn_bbb_2080ti.json": "bbb2080",
    "../data/bbb/imdn_div2k_3060ti.json": "div3060",
    "../data/bbb/imdn_bbb_3060ti.json": "bbb3060",
}


def generate_table():
    """
    Generate automatic parameter tuning table for BONES
    :return:
    """
    global target_name

    # pre-defined parameters
    np.random.seed(0)
    num_bandwidth = 25
    min_bandwidth = 200
    step_bandwidth = 200
    num_variance = 10
    min_variance = 500
    step_variance = 500
    num_latency = 2
    min_latency = 20
    step_latency = 80
    num_trace = 20
    num_period = 159 * 4
    size_period = 1000

    # generate synthetic network traces
    input_list = []
    for i in range(num_bandwidth):
        for j in range(num_variance):
            for k in range(num_latency):
                bandwidth = np.random.normal(min_bandwidth + i * step_bandwidth, min_variance + i * step_variance, (num_trace, num_period))
                bandwidth[bandwidth < 0] = 0
                latency = (min_latency + step_latency * k) * np.ones((num_period))
                net_time = size_period * np.ones((num_period))
                input_list.append((i, j, k, bandwidth, latency, net_time))

    # search for optimal parameters
    results = []
    pool_size = cpu_count()  # number of workers in the pool
    with Pool(pool_size) as p:
        with tqdm(total=len(input_list)) as pbar:
            for i, result in tqdm(enumerate(p.imap_unordered(search_bones, input_list))):
                pbar.update()
                results.append(result)

    # record optimal parameters
    gamma_p_table = -1 * np.ones((num_bandwidth, num_variance, num_latency))
    V_multiplier_table = -1 * np.ones((num_bandwidth, num_variance, num_latency))
    for result in results:
        idx_mean, idx_var, idx_lat, gamma_p, V_multiplier, qoe = result
        gamma_p_table[idx_mean, idx_var, idx_lat] = gamma_p
        V_multiplier_table[idx_mean, idx_var, idx_lat] = V_multiplier
    print("gamma_p_table: {}".format(gamma_p_table))
    print("V_multiplier_table: {}".format(V_multiplier_table))
    np.savez(f"autotune_bones_{target_name}.npz", gamma_p_table=gamma_p_table, V_multiplier_table=V_multiplier_table)
    return


def search_bones(inputs):
    """
    Grid search for optimal parameters of BONES
    :param inputs: network condition indices, synthetic network traces
    :return:
    """
    idx_mean, idx_var, idx_lat, bandwidth, latency, net_time = inputs

    # parameter search space
    gamma_p_list = [10 + 10 * i for i in range(10)]
    V_multiplier_list = [0.1 + 0.1 * i for i in range(10)]

    # grid search
    result = np.zeros((len(gamma_p_list), len(V_multiplier_list)))
    for i in range(len(gamma_p_list)):
        for j in range(len(V_multiplier_list)):
            gamma_p = gamma_p_list[i]
            V_multiplier = V_multiplier_list[j]
            result[i, j] = run_bones(gamma_p, V_multiplier, bandwidth, latency, net_time)

    i_best = np.argmax(result) // len(V_multiplier_list)
    j_best = np.argmax(result) % len(V_multiplier_list)
    gamma_p = gamma_p_list[i_best]
    V_multiplier = V_multiplier_list[j_best]
    qoe = result[i_best, j_best]
    result = (idx_mean, idx_var, idx_lat, gamma_p, V_multiplier, qoe)
    return result


def run_bones(gamma_p, V_multiplier, bandwidth, latency, net_time, num_trace=20):
    parser = argparse.ArgumentParser(description='Simulator for neural-enhanced streaming system.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_path', default="",
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
    parser.add_argument('--download_buffer', default=60, type=int,
                        help='Maximum download buffer size, measured in seconds.')
    parser.add_argument('--enhance_buffer', default=60, type=int,
                        help='Maximum enhance buffer size, measured in seconds.')
    parser.add_argument('--bandwidth_multiplier', default=1, type=float,
                        help='Multiplier for the network bandwidth.')
    parser.add_argument('--speed_multiplier', default=1, type=float,
                        help='Multiplier for the enhancement speed.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose information.')

    parser.add_argument('--gamma_p', type=float, default=10,
                        help='BONES gamma*p parameter.')
    parser.add_argument('--V_multiplier', type=float, default=1,
                        help='BONES gamma*p parameter.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor download process.')
    parser.add_argument('--autotune', action='store_true', default=False,
                        help='Automatic parameter tuning.')

    args = parser.parse_args()

    # overwrite algorithm parameters
    global target_name
    args.enhance_path = name2path[target_name]
    args.gamma_p = gamma_p
    args.V_multiplier = V_multiplier

    avg_qoe = 0
    for i in range(num_trace):
        simulator = BONES(args)
        # overwrite network traces
        simulator.network.net_time = net_time
        simulator.network.bandwidth = bandwidth[i, :]
        simulator.network.latency = latency
        simulator.network.num_period = net_time.shape[0]

        simulator.run()

        result = simulator.report()
        avg_qoe += result["qoe"]

    avg_qoe /= num_trace
    return avg_qoe


if __name__ == '__main__':
    target_names = ["nas", "div2080", "bbb2080", "div3060", "bbb3060"]
    for tm in target_names:
        target_name = tm
        print("target_name: {}".format(target_name))
        generate_table()