import gzip
import matplotlib.pyplot as plt
import os
import pathlib
import numpy as np

from base import BaseDataset
"""
    This file contains all your datasets for this project.
    Add a class of your dataset.
    Inherit the BaseDataset class.
"""
class Fruit(BaseDataset):
    """
        Fruit dataset for DL-HW3
    """
    def __init__(self, data_dir, train):
        super().__init__(data_dir, train)

    def _extract_training_data(self, data_dir):
        train_path = '.' + data_dir + 'Data_train'
        labels = os.listdir(train_path)
        X_train = []
        y_train = []
        # Loop through the three labels
        for dir in labels:
            for path in os.listdir(train_path + '/' + dir):
                img_path = train_path + '/' + dir + '/' + path
                # The image is in RGBA form
                img = plt.imread(img_path)
                # Remove the alpha channel
                img = img[:, :, 0:3]
                X_train.append(img)
                # Deal with the labels
                if dir == labels[0]:
                    y_train.append(0)
                if dir == labels[1]:
                    y_train.append(1)
                if dir == labels[2]:
                    y_train.append(2)
        # PNG file is already in float, 0-1, no need to normalize
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        return X_train, y_train
    def _extract_test_data(self, data_dir):
        train_path = '.' + data_dir + 'Data_test'
        labels = os.listdir(train_path)
        X_test = []
        y_test = []
        # Loop through the three labels
        for dir in labels:
            for path in os.listdir(train_path + '/' + dir):
                img_path = train_path + '/' + dir + '/' + path
                # The image is in RGBA form
                img = plt.imread(img_path)
                # Remove the alpha channel
                img = img[:, :, 0:3]
                X_test.append(img)
                # Deal with the labels
                if dir == labels[0]:
                    y_test.append(0)
                if dir == labels[1]:
                    y_test.append(1)
                if dir == labels[2]:
                    y_test.append(2)
        # PNG file is already in float, 0-1, no need to normalize
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        return X_test, y_test
