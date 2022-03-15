import numpy as np
from dataset import datasets
from base import BaseDataLoader

"""
    This file will contain all the dataloaders for all the datasets
    inside your project
"""
class FruitDataLoader(BaseDataLoader):
    """
        A data loader for the fruit dataset
        Labels:
            (1) Carambola
            (2) Lychee
            (3) Pear
    """
    def __init__(self, data_dir, batch_size, shuffle=True, val_split=0.0, n_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = datasets.Fruit(self.data_dir, train=training)
        super().__init__(self.dataset, batch_size, shuffle, val_split, n_workers)
        # if training, val_split != 0
        if self.val_split != 0.0:
            self.X_train, self.y_train, self.X_val, self.y_val = self._preprocess_train_val(self.X_train, self.y_train, self.X_val, self.y_val)
            print("X_train shape:", self.X_train.shape)
            print("y_train shape:", self.y_train.shape)
            print("X_val shape:", self.X_val.shape)
            print("y_val shape:", self.y_val.shape)
        # Else, it is testing
        else:
            self.X_test, self.y_test = self._preprocess_test()
    def _preprocess_test(self):
        """
            Preprocess your test data to the correct shape.
            Perform one-hot encoding
        """
        y_test = self.dataset.y_test
        X_test = self.dataset.X_test
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[3], X_test.shape[1], X_test.shape[2]))
        len_test = len(y_test)

        # One hot encoding
        labels = 3
        y_test = y_test.reshape(1, len_test)
        y_test = np.eye(labels)[y_test.astype('int32')]
        # Adjust the shape
        y_test = y_test.reshape(len_test, labels)

        return X_test, y_test

    def _preprocess_train_val(self, X_train, y_train, X_val, y_val):
        """
            Handle training and validation data for training.
        """
        # Transpose the data
        y_train = y_train
        y_val = y_val

        X_train = np.array(X_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[3], X_train.shape[1], X_train.shape[2]))
        X_val = np.array(X_val)
        X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[3], X_val.shape[1], X_val.shape[2]))

        # Basic infos
        labels = 3
        len_train = len(y_train)
        len_val = len(y_val)
        # One-hot encoding
        y_train = y_train.reshape(1, len_train)
        y_val = y_val.reshape(1, len_val)
        y_train = np.eye(labels)[y_train.astype('int32')]
        y_val = np.eye(labels)[y_val.astype('int32')]
        
        # Adjust the shape
        y_train = y_train.reshape(len_train, labels)
        y_val = y_val.reshape(len_val, labels)


        return X_train, y_train, X_val, y_val
