import argparse

class NASConfig():
    def __init__(self):
        pass

    def add_args(self, parser):
        parser.add_argument('--model_path', default='../rl/nas/actor_logb_10k.pt',
                            help='Specify the .json file describing the video manifest.')
        parser.add_argument('--deadline_calc', default='min_down_seg_enh',
                            help='Method used to calculate the playback deadline of a video chunk.')
        parser.add_argument('-i', '--impatient', action='store_true',
                            help='Adopt impatient enhancement.')
        args = parser.parse_args()
        return args
