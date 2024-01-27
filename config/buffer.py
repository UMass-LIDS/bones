import argparse

class BufferConfig():
    def __init__(self):
        pass

    def add_args(self, parser):
        parser.add_argument('-r', '--reservoir', default=0.375,
                            help='Buffer-based algorithm bandwidth reservoir level.')
        parser.add_argument('-ur', '--upper_reservoir', default=0.9,
                            help='Buffer-based algorithm bandwidth upper reservoir level.')
        parser.add_argument('-i', '--impatient', action='store_true',
                            help='Adopt impatient enhancement.')
        args = parser.parse_args()
        return args