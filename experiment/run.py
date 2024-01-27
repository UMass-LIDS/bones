import copy
import sys
sys.path.append("..")
import argparse
import os
import config
import control
from multiprocessing import Pool, cpu_count
import numpy as np
import random

# do not test over these datasets
exclude_dirs = ["pensieve", "fcc_random", "puffer", "fcc_10k"]

# exclude traces with average bandwidth lower than 400 kbps
exclude_names = ["report.2010-09-14_1415CEST.json", "report.2011-02-01_0840CET.json", "report.2011-02-01_1000CET.json"]

def test(args):
    np.random.seed(0)
    random.seed(0)

    dir_names = os.listdir(args.trace_root)
    cnt = 0
    final_results = {}
    input_list = []

    for dir_name in dir_names:
        if dir_name in exclude_dirs:
            continue

        dir_path = os.path.join(args.trace_root, dir_name)
        if not os.path.isdir(dir_path):
            continue

        final_results[dir_name] = []
        trace_names = os.listdir(dir_path)
        for trace_name in trace_names:
            if trace_name in exclude_names:
                continue

            trace_path = os.path.join(dir_path, trace_name)
            new_args = copy.deepcopy(args)
            new_args.trace_path = trace_path
            input_list.append((dir_name, new_args))


    pool_size = cpu_count()
    with Pool(pool_size) as p:
        for i, result in enumerate(p.imap_unordered(run_simulator, input_list)):
            dir_name, report = result
            final_results[dir_name].append(report)
            cnt += 1

    with open(args.log_path, 'w') as file:
        dir_report = []
        for dir_name in final_results:
            report = aggregate(final_results[dir_name])
            summarize(file, "Report: " + dir_name, report)
            dir_report.append(report)
        final_report = aggregate(dir_report)
        summarize(file, "Report: All", final_report)
    return


def run_simulator(inputs):
    dir_name, args = inputs
    method_class = getattr(control, args.method)
    simulator = method_class(args)
    simulator.run()
    report = simulator.report()
    return dir_name, report

def aggregate(results):
    report = {
        "average quality vmaf": 0,
        "average oscillation vmaf": 0,
        "average quality ssim": 0,
        "average oscillation ssim": 0,
        "freeze time": 0,
        "average bandwidth": 0,
        "qoe": 0,
    }
    for result in results:
        for key in report:
            report[key] += result[key]
    for key in report:
        report[key] /= len(results)
    report["average bandwidth"] = float(report["average bandwidth"])
    return report


def summarize(file, title, info):
    msg = ""
    msg += title + " \n"
    for key in info:
        msg += "{}: {} \n".format(key, info[key])
    file.write(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator for neural-enhanced streaming system.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', required=True, type=str,
                        help='Method name.')
    parser.add_argument('--log_path', required=True, type=str,
                        help='Path the logging file.')
    parser.add_argument('--trace_root', default='../data/trace/',
                        help='Specify the root directory for network traces.')
    parser.add_argument('--manifest_path', default='../data/bbb/bbb_video_info.json',
                        help='Specify the .json file describing the video manifest.')
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
    parser.add_argument('--know_seg', action='store_true',
                        help='Has knowledge about the quality of each segment instead of an average.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose information.')
    args = parser.parse_known_args()[0]

    # add method-specific arguments
    config_class = getattr(config, args.method + "Config")()
    args = config_class.add_args(parser)
    # start testing
    test(args)



