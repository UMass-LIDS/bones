import os
import json

def processFile(tracePath):
    converted_json = []
    with open(tracePath, 'r') as f:
        lines = f.readlines()
        numLines = len(lines)
        prev_timestamp = float(lines[0].split()[0].strip())
        for lineNum in range(1, numLines):
            components = lines[lineNum].split()
            curr_timestamp = float(components[0].strip())
            mbitspersec = float(components[1].strip())
            json_obj = {"duration_ms": (curr_timestamp - prev_timestamp)*1000.0, "bandwidth_kbps": mbitspersec*1000.0, "latency_ms": 100}
            prev_timestamp = curr_timestamp
            converted_json.append(json_obj)
    return converted_json

def iterate_over_logs():
    logPaths = [f for f in os.listdir(ROOT_DIR) if "norway" in f]
    for logPath in logPaths:
        converted_json = processFile(os.path.join(ROOT_DIR, logPath))
        save_path = os.path.join(PROCESSED_DIR, logPath) + ".json"
        with open(save_path, "w") as f_w:
            f_w.write(json.dumps(converted_json))

ROOT_DIR = "./raw_logs"
PROCESSED_DIR = "./processed_logs"
iterate_over_logs()   
