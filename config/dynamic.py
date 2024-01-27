import argparse

class DynamicConfig():
    def __init__(self):
        pass

    def add_args(self, parser):
        parser.add_argument('-sw', '--switch', default=10000, type=float,
                            help='Buffer level threshold for algorithm swtiching.')
        parser.add_argument('-gp', '--gamma_p', type=float, default=5,
                            help='BOLA gamma_p parameter, measured in seconds.')
        parser.add_argument('-s', '--safety', default=0.9, type=float,
                            help='Throughput-based algorithm bandwidth safety factor.')
        parser.add_argument('-m', '--monitor', action='store_true',
                            help='Monitor download process.')
        parser.add_argument('-i', '--impatient', action='store_true',
                            help='Adopt impatient enhancement.')
        args = parser.parse_args()
        return args