import os
import json

def processBelgiumFile(tracePath):
    converted_json = []
    with open(tracePath, 'r') as f:
        lines = f.readlines()
        numLines = len(lines)
        prev_timestamp = float(lines[0].split()[0].strip())
        for lineNum in range(1, numLines):
            components = lines[lineNum].split()
            curr_timestamp = float(components[1].strip())
            bytes_recv = float(components[4].strip())
            recv_time = float(components[5].strip())
            kbps = (bytes_recv / recv_time)*8.0
            json_obj = {"duration_ms": (curr_timestamp - prev_timestamp), "bandwidth_kbps": kbps, "latency_ms": 20}
            prev_timestamp = curr_timestamp
            converted_json.append(json_obj)
    return converted_json

def iterate_over_logs():
    logPaths = [f for f in os.listdir(ROOT_DIR) if "report" in f]
    for logPath in logPaths:
        converted_json = processBelgiumFile(os.path.join(ROOT_DIR, logPath))
        save_path = os.path.join(PROCESSED_DIR, logPath) + ".json"
        with open(save_path, "w") as f_w:
            f_w.write(json.dumps(converted_json))

ROOT_DIR = "./raw_logs"
PROCESSED_DIR = "./processed_logs"
iterate_over_logs()   
