import collections
import sys
import json
import numpy as np


class Logger(object):
    """tee functionality in python. If this object exists,
    then all of stdout gets logged to the file
    Adoped from:
    https://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python/3423392#3423392
    """

    def __init__(self, name, mode='a'):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        # flush right away
        self.file.flush()
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()


class NumpyAwareJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


def write_json(obj, fname, **kwargs):
    with open(fname, "w") as f:
        return json.dump(obj, f, cls=NumpyAwareJSONEncoder, **kwargs)


def prefix_dict(d, prefix=''):
    return {prefix + k: v for k, v in d.items()}
