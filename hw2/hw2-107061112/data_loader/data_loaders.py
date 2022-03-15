import torch
from base import BaseDataLoader
from dataset import datasets
from torchvision import transforms

class WaferDataLoader(BaseDataLoader):
    """
        A custom data loader for the wafer dataset
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = datasets.WaferData(self.data_dir, train=training, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        self.label_name = self.dataset.classes
