import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from pathlib import Path


def read_json(fname):
    """ 
        A utility to read in a json config file
    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)
def plot_learning_curve(string, step, data, x):
    """
        Plot the accuracy curve and loss curve wrt epochs
    """
    fig = plt.figure(num=0)
    plt.xlabel('Epoch')
    if string == 'Loss':
        plt.plot(step, data['Training loss'], label='training')
        plt.plot(step, data['Validation loss'], label='validation')
    else:
        plt.plot(step, data['Training acc'], label='training')
        plt.plot(step, data['Validation acc'], label='validation')
    fig.suptitle(string + "\n", fontsize=16)
    #  plt.axis([1, x, 0, 1])
    plt.legend(loc='lower right')
    #  plt.show()
    plt.savefig(fname=string)
    plt.close(fig)


class MetricTracker:
    """
        A metric tracker class to handle all the metrics you give
    """
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
