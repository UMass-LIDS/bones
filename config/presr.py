import argparse

class PreSRConfig():
    def __init__(self):
        pass

    def add_args(self, parser):
        parser.add_argument('-la', '--look_ahead', default=5, type=int,
                            help='Look-ahead window size.')
        parser.add_argument('-qf', '--quality_factor', default=10, type=int,
                            help='Penalty factor for quality.')
        parser.add_argument('-vf', '--variation_factor', default=10, type=int,
                            help='Penalty factor for quality variation.')
        parser.add_argument('-rf', '--rebuffer_factor', default=1, type=int,
                            help='Penalty factor for rebuffering events.')
        args = parser.parse_args()
        return args