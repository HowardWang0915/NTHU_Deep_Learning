import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def imshow(mother_img, decode_img, class_name, vis=False):
    """
        A function to draw the mother image and the 
        generated image.
    """
    fig, axes = plt.subplots(1, 6, figsize=(15, 5))
    if not vis:
        mother_img = mother_img.cpu().numpy()
    for i in range(6):
        if i == 0:
            axes[i].set_title("Original:" + class_name, fontsize=18)
            if not vis:
                axes[i].imshow(np.argmax(np.transpose(mother_img, (1, 2, 0)), axis=2))
            else:
                axes[i].imshow(np.argmax(np.transpose(mother_img, (0, 1, 2)), axis=2))
        else:
            if not vis:
                decode_img[i-1] = decode_img[i-1].cpu().numpy()
            axes[i].set_title('Generated ' + str(i), fontsize=18)
            if not vis:
                axes[i].imshow(np.argmax(np.transpose(decode_img[i-1], (1, 2, 0)), axis=2))
            else:
                axes[i].imshow(np.argmax(np.transpose(decode_img[i-1], (0, 1, 2)), axis=2))
    fig.tight_layout()
    if not vis:
        plt.savefig('img/' + class_name + '.png')
    else:
        plt.show()
    plt.close()
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
