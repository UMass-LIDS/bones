import sys
import os

class Logger(object):
    def __init__(self, file_path="", verbose=False, get_timestamp=None):
        """
        Log information to console and file
        :param file_path: path to the logging file
        :param verbose: print verbose information
        :param get_timestamp: function handle that returns a timestamp
        """
        self.file = None
        self.verbose = verbose
        self.get_timestamp = get_timestamp

        if file_path != "":
            self.file = open(file_path, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def summarize(self, title, info):
        msg = ""
        msg += title + " \n"
        for key in info:
            msg += "{}: {} \n".format(key, info[key])
        print(msg)
        if self.file is not None:
            self.file.write(msg)

    def write(self, msg):
        if self.get_timestamp:
            timestamp = self.get_timestamp()
            msg = timestamp + msg + "\n"
        if self.verbose:
            print(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.file is not None:
            self.file.close()

    def is_verbose(self):
        return self.verbose
