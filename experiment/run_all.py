import copy
import subprocess as sp
import itertools
from collections import OrderedDict
import os

# base_dict = OrderedDict({
#     "enhance": ["nas", "bbb2080", "bbb3060"]
# })

base_dict = OrderedDict({
    # "enhance": ["nas", "div2080", "bbb2080", "div3060", "bbb3060"]  # enhance setting
    "enhance": ["bbb3060"]  # enhance setting
    # "bm": [1],  # bandwidth multiplier
    # "sm": [1],  # speed multiplier
    # "db": [60],  # download buffer
    # "eb": [60],  # enhance buffer
})


bola_dict = OrderedDict({
    "method": ["BOLA"],
    "gp": [5],  # gamma_p
    "m": [True],  # monitor
    "i": [False, True],  # impatient
})


# bones_dict = OrderedDict({
#     "method": ["BONES"],
#     "gp": [10],  # gamma_p
#     "vm": [1],  # V_multiplier
#     "m": [True],  # monitor
#     "a": [True],  # autotune
# })

bones_dict = OrderedDict({
    "method": ["BONES"],
    "gp": [10],  # gamma_p
    "vm": [1],  # V_multiplier
    "m": [True],  # monitor
    "a": [True],  # autotune
    "q": [True],  # quick_start
})

mpc_dict = OrderedDict({
    "method": ["MPC"],
    "qf": [10],  # quality_factor
    "vf": [10],  # variation_factor
    "rf": [1],  # rebuffer_factor
    "i": [False, True],  # impatient
})

buffer_dict = OrderedDict({
    "method": ["Buffer"],
    "i": [False, True],  # impatient
})

tput_dict = OrderedDict({
    "method": ["Throughput"],
    "s": [0.9],  # safety
    "i": [False, True],  # impatient
})

presr_dict = OrderedDict({
    "method": ["PreSR"],
    "qf": [10],  # quality_factor
    "vf": [10],  # variation_factor
    "rf": [1],  # rebuffer_factor
})

dynamic_dict = OrderedDict({
    "method": ["Dynamic"],
    # "sw": [10000], # switch
    "s": [0.9],  # safety
    "gp": [5],  # gamma_p
    "m": [False, True],  # monitor
    "i": [False, True],  # impatient
})

pensieve_dict = OrderedDict({
    "method":["Pensieve"],
    "i": [False, True],  # impatient
})

nas_dict = OrderedDict({
    "method":["NAS"]
})

bones_vary_dict = OrderedDict({
    "method": ["BONESVary"],
    "gp": [10],  # gamma_p
    "vm": [1],  # V_multiplier
    "m": [True],  # monitor
    "a": [True],  # autotune
    # "nm": [1],  # noise_mean
    # "nr": [0.1 + 0.1 * i for i in range(20)],  # noise_range
    "nm": [0.1 * i for i in range(21)],  # noise_mean
    "nr": [0.3],  # noise_range
})


def run(method_dict):
    merge_dict = OrderedDict({**method_dict, **base_dict})
    combination = list(itertools.product(*merge_dict.values()))
    for comb in combination:
        args_dict = dict(zip(merge_dict.keys(), comb))
        log_path = f"{args_dict['enhance']}_logs/"
        os.makedirs(log_path, exist_ok=True)
        command = "python run.py"

        for arg_name in args_dict:
            arg_value = args_dict[arg_name]
            if arg_name == "method":
                log_path += f"{arg_value}"
                command += f" --{arg_name} {arg_value}"
            elif arg_name == "enhance":
                name2path = {
                    "nas": "../data/bbb/nas_1080ti.json",
                    "div2080": "../data/bbb/imdn_div2k_2080ti.json",
                    "bbb2080": "../data/bbb/imdn_bbb_2080ti.json",
                    "div3060": "../data/bbb/imdn_div2k_3060ti.json",
                    "bbb3060": "../data/bbb/imdn_bbb_3060ti.json",
                }
                command += f" --enhance_path {name2path[arg_value]}"
            elif isinstance(arg_value, bool):
                if arg_value:
                    log_path += f"_{arg_name}"
                    command += f" -{arg_name}"
            elif isinstance(arg_value, float):
                log_path += f"_{arg_name}{arg_value:.2f}"
                command += f" -{arg_name} {arg_value}"
            else:
                log_path += f"_{arg_name}{arg_value}"
                command += f" -{arg_name} {arg_value}"

        log_path += ".txt"
        command += f" --log_path {log_path}"

        print("Generating: ", log_path)
        # print("Command: ", command)
        sp.run(command, shell=True, text=True)
    return


def run_all():
    # run(bola_dict)
    run(bones_dict)
    # run(mpc_dict)
    # run(buffer_dict)
    # run(tput_dict)
    # run(presr_dict)
    # run(dynamic_dict)
    # run(pensieve_dict)
    # run(nas_dict)
    return


if __name__ == '__main__':
    run_all()
