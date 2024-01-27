import argparse

class BONESVaryConfig():
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
        parser.add_argument('-nm', '--noise_mean', type=float, default=1,
                            help='Computation speed noise mean.')
        parser.add_argument('-nr', '--noise_range', type=float, default=0.1,
                            help='Computation speed noise range.')
        args = parser.parse_args()
        return args