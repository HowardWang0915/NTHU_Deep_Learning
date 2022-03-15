import zipfile
import torch
import numpy as np
from torch.utils.data import Dataset

class WaferData(Dataset):
    """
        Classes:
        (0). Center
        (1). Donut
        (2). Edge-Loc
        (3). Edge-Ring
        (4). Loc
        (5). Near-full
        (6). Random
        (7). Scratch
        (8). None
    """
    def __init__(self, data_root, train, transform=None): 
        self.data_root = data_root
        self.transform = transform
        self.train = train
        self.data, self.label = self.__load_data()
        self.classes = [
            'Center',
            'Donut',
            'Edge-Loc',
            'Edge-Ring',
            'Loc',
            'Near-full',
            'Random',
            'Scratch',
            'None'
        ]

    def __load_data(self):
        data = {}
        file = self.data_root + 'wafer.zip'
        with zipfile.ZipFile(file) as zf:
            for file_name in zf.namelist():
                dataset = np.load(file)
                data[file_name] = dataset[file_name]
            return data['data.npy'], data['label.npy']
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label
    
    def __len__(self):
        return len(self.data)
