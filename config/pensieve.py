import argparse

class PensieveConfig():
    def __init__(self):
        pass

    def add_args(self, parser):
        parser.add_argument('--model_path', default='../rl/pensieve/actor_logb_10k.pt',
                            help='Specify the .json file describing the video manifest.')
        parser.add_argument('-i', '--impatient', action='store_true',
                            help='Adopt impatient enhancement.')
        args = parser.parse_args()
        return args