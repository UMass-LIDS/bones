import argparse

class LeiaNoneConfig():
    def __init__(self):
        pass

    def add_args(self, parser):
        parser.add_argument('--look_ahead', default=5, type=int,
                            help='Look-ahead window size.')
        parser.add_argument('--max_change', default=4, type=int,
                            help='Maximum bitrate level change.')
        parser.add_argument('--rebuffer_factor', default=2500,
                            help='Penalty factor for rebuffering events.')
        parser.add_argument('--switch_factor', default=2500,
                            help='Penalty factor for bitrate switches.')
        args = parser.parse_args()
        return args