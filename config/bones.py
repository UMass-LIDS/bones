import argparse

class BONESConfig():
    def __init__(self):
        pass

    def add_args(self, parser):
        parser.add_argument('-gp', '--gamma_p', type=float, default=10,
                            help='BONES gamma*p parameter.')
        parser.add_argument('-vm', '--V_multiplier', type=float, default=1,
                            help='BONES gamma*p parameter.')
        parser.add_argument('-m', '--monitor', action='store_true',
                            help='Monitor download process.')
        parser.add_argument('-a', '--autotune', action='store_true',
                            help='Automatic parameter tuning.')
        parser.add_argument('-q', '--quick_start', action='store_true',
                            help='Quick start.')
        args = parser.parse_args()
        return args