import numpy as np
import tensorflow as tf
import tflearn
import json, pickle

STATE_LEN = 6 # Number of state parameters
STATE_MEMORY_LEN = 8 # Number of frames to consider from the past
ACTION_LEN = 6 # Number of possible discrete actions (bit-rates of the next chunk)

def saveAsCSV(objToWrite, fName):
    stringToWrite = ""
    for key in objToWrite:
        if isinstance(objToWrite[key], (np.ndarray, np.generic)):
            stringToWrite += str(key) + ","
            flatArr = objToWrite[key].flatten()
            for elem in flatArr:
                stringToWrite += str(elem) + ","
            stringToWrite = stringToWrite[:-1]
            stringToWrite += '\n'
            print(objToWrite[key].shape)
        else:
            stringToWrite += str(key) + "," + '"' +str(objToWrite[key]) + '"' + "\n"
    with open(fName, 'w') as f:
        f.write(stringToWrite)


def saveAsPickle(objToWrite, fName):
    filehandler = open(fName, 'w')
    pickle.dump(objToWrite, filehandler)

def saveAsJSON(objToWrite, fName):
    json_str = json.dumps(objToWrite)
    with open(fName, "w") as f:
        f.write(json_str)
    pass

# load the TF checkpoint model
NN_MODEL = './pretrain_linear_reward.ckpt'

reader = tf.train.load_checkpoint(NN_MODEL)

# Parse weights from the checkpoint
dtype_map  = reader.get_variable_to_dtype_map()
shape_map  = reader.get_variable_to_shape_map()
print(shape_map)
state_dict = {v: reader.get_tensor(v) for v in shape_map}
print(state_dict)

inputs = tflearn.input_data(shape=[None, STATE_LEN, STATE_MEMORY_LEN])

split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], 128, activation='relu')
split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], 128, activation='relu')
split_2 = tflearn.conv_1d(inputs[:, 2:3, :], 128, 4, activation='relu')
split_3 = tflearn.conv_1d(inputs[:, 3:4, :], 128, 4, activation='relu')
split_4 = tflearn.conv_1d(inputs[:, 4:5, :ACTION_LEN], 128, 4, activation='relu')
split_5 = tflearn.fully_connected(inputs[:, 4:5, -1], 128, activation='relu')

split_2_flat = tflearn.flatten(split_2)
split_3_flat = tflearn.flatten(split_3)
split_4_flat = tflearn.flatten(split_4)

merge_net = tflearn.merge([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')

dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
out = tflearn.fully_connected(dense_net_0, ACTION_LEN, activation='softmax')

state_dict["is_training"] = bool(state_dict["is_training"])
saveAsCSV(state_dict, 'state.csv')
saveAsCSV(shape_map, 'shape.csv')

#
#
