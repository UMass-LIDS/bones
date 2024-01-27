import argparse

class ThroughputConfig():
    def __init__(self):
        pass

    def add_args(self, parser):
        parser.add_argument('-s', '--safety', default=0.9, type=float,
                            help='Throughput-based algorithm bandwidth safety factor.')
        parser.add_argument('-m', '--monitor', action='store_true',
                            help='Monitor download process.')
        parser.add_argument('-i', '--impatient', action='store_true',
                            help='Adopt impatient enhancement.')
        args = parser.parse_args()
        return args