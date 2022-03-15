import gzip
import pathlib
import numpy as np

from base import BaseDataset
"""
    This file contains all your datasets for this project.
    Add a class of your dataset.
    Inherit the BaseDataset class.
"""
class MNIST(BaseDataset):
    """
        MNIST dataset
    """
    def __init__(self, data_dir, train):
        super().__init__(data_dir, train)

    def _extract_training_data(self, data_dir):
        # 1. First, read in the training images
        f = gzip.open(str(pathlib.Path().resolve()) + data_dir + 'train-images-idx3-ubyte.gz', 'r')
        # Consume the magic number
        f.read(4)
        # How many image are there?
        image_count = int.from_bytes(f.read(4), 'big')
        # Image height
        image_height = int.from_bytes(f.read(4), 'big')
        # Image width
        image_width = int.from_bytes(f.read(4), 'big')
        # Read in the raw image data
        image_data = f.read()
        # image data to numpy array
        images = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)
        X_train = images.reshape(image_count, image_height * image_width)

        # 2. Then, read in the training labels
        f = gzip.open(str(pathlib.Path().resolve()) + data_dir + 'train-labels-idx1-ubyte.gz', 'r')
        # Consume the magic number
        f.read(4)

        # Label count
        label_count = int.from_bytes(f.read(4), 'big')

        # The rest are the actual raw label data
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        # labels data to numpy array
        y_train = labels.reshape(-1, 1)

        return X_train, y_train


    def _extract_test_data(self, data_dir):
        """ 
            A helper function to read in the dataset
            :data_dir: the directory of the data
        """
        # 1. Read in the testing images
        f = gzip.open(str(pathlib.Path().resolve()) + data_dir + 't10k-images-idx3-ubyte.gz', 'r')

        # Consume the magic number
        f.read(4)
        # How many image are there?
        image_count = int.from_bytes(f.read(4), 'big')
        # Image height
        image_height = int.from_bytes(f.read(4), 'big')
        # Image width
        image_width = int.from_bytes(f.read(4), 'big')
        # read in raw image data
        image_data = f.read()
        # image data to numpy array
        images = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)
        X_test = images.reshape(image_count, image_height * image_width)


        # 2. Then, read in the testing labels
        f = gzip.open(str(pathlib.Path().resolve()) + data_dir + 't10k-labels-idx1-ubyte.gz', 'r')
        # Consume the magic number
        f.read(4)

        # Label count
        label_count = int.from_bytes(f.read(4), 'big')

        # The rest are the actual raw label data
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        # labels data to numpy array
        y_test = labels.reshape(-1, 1)

        return X_test, y_test
