# column 0 = uid
# column 1 = timestamp
# column 2 = target
# column 3 = address
import json

import numpy as np
import datetime
import os
import random


# FILE_PATH = '../../data/trace/pensieve/data-raw-2016-jun/201606/curr_webget_2016_06.csv'
# OUTPUT_PATH = './3Gtraces/'

FILE_PATH = 'D:/Datasets/FCC/data-raw-2016-jun/201606/curr_webget_2016_06.csv'
# OUTPUT_PATH = 'D:/Datasets/FCC/data-raw-2016-jun/201606/fcc_random/'
OUTPUT_PATH = 'D:/Datasets/FCC/data-raw-2016-jun/201606/fcc_10k/'
CONSTANT_LATENCY = 100 # in milliseconds
CONSTANT_DURATION = 5000 # in milliseconds
NUM_LINES = np.inf
TIME_ORIGIN = datetime.datetime.utcfromtimestamp(0)


if not os.path.exists(OUTPUT_PATH):
	os.mkdir(OUTPUT_PATH)

bw_measurements = {}
def processFile(file_path):
	line_counter = 0
	with open(file_path, 'r') as f:
		for line in f:
			parse = line.split(',')

			uid = parse[0]
			dtime = (datetime.datetime.strptime(parse[1],'%Y-%m-%d %H:%M:%S')
				- TIME_ORIGIN).total_seconds()
			target = parse[2]
			address = parse[3]
			throughput = parse[6]  # bytes per second

			k = (uid, target)
			if k in bw_measurements:
				bw_measurements[k].append(throughput)
			else:
				bw_measurements[k] = [throughput]

			line_counter += 1
			print(str(line_counter))
			if line_counter >= NUM_LINES:
				break

	for k in bw_measurements:
		out_file = 'trace_' + '_'.join(k)
		out_file = out_file.replace(':', '-')
		out_file = out_file.replace('/', '-')
		out_file = OUTPUT_PATH + out_file
		with open(out_file, 'w') as f:
			for i in bw_measurements[k]:
				f.write(str(i) + '\n')

def convert_trace(fPath):
	converted = []
	with open(fPath, 'r') as f:
		lines = f.readlines()
		for line in lines:
			bitrate = float(line)*8/1000.0 # Convert bytes per second into kilobits per second
			converted.append({"bandwidth_kbps": bitrate, "duration_ms": CONSTANT_DURATION,"latency_ms": CONSTANT_LATENCY})
	return converted

def check_trace(converted, threshold=400):
	total_bandwdith = np.zeros((1,), dtype=np.float64)
	total_duration = np.zeros((1,), dtype=np.float64)
	for i in range(len(converted)):
		element = converted[i]
		total_bandwdith += element["bandwidth_kbps"] * element["duration_ms"]
		total_duration += element["duration_ms"]
	if total_bandwdith/total_duration < threshold:
		return False
	print("Average bandwidth: ", total_bandwdith/total_duration)
	return True


def pick_traces():
	traces = [f for f in os.listdir(OUTPUT_PATH)]
	num_traces = len(traces)
	# num_to_pick = 1000
	num_to_pick = 10000
	pool = []
	cnt = 0
	while cnt < num_to_pick:
			index = random.randrange(0, num_traces)
			if index in pool:
				continue
			trace = traces[index]
			convertedJSON = convert_trace(OUTPUT_PATH + trace)
			if check_trace(convertedJSON):
				# converted_json_path = "../../data/trace/fcc_random/" + trace + ".json"
				converted_json_path = "../../data/trace/fcc_10k/" + trace + ".json"
				with open(converted_json_path, "w") as f:
					f.write(json.dumps(convertedJSON))
				cnt += 1
				pool.append(index)


if __name__ == '__main__':
	# processFile(FILE_PATH)
	pick_traces()