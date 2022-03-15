import numpy as np
from dataset import datasets
from base import BaseDataLoader

"""
    This file will contain all the dataloaders for all the datasets
    inside your project
"""
class MNISTDataLoader(BaseDataLoader):
    """
        MNIST data loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, val_split=0.0, n_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training)
        #  import pdb; pdb.set_trace()
        super().__init__(self.dataset, batch_size, shuffle, val_split, n_workers)
        # Preprocess the dataset
        if self.val_split != 0.0:
            self.X_train, self.y_train, self.X_val, self.y_val = self._preprocess_train_val(self.X_train, self.y_train, self.X_val, self.y_val)
            print("X_train shape:", self.X_train.shape)
            print("y_train shape:", self.y_train.shape)
            print("X_val shape:", self.X_val.shape)
            print("y_val shape:", self.y_val.shape)
        else:
            self.X_test, self.y_test = self._preprocess_test()


    def _preprocess_test(self):
        """
            Preprocess your test data to the correct shape.
            Also, perform normalization on the data, as well as 
            one-hot encoding
        """
        y_test_T = self.dataset.y_test.T 
        X_test_T = self.dataset.X_test.T 

        # One hot encoding
        digits = 10
        y_test_T = np.eye(digits)[y_test_T.astype('int32')]
        y_test_T = y_test_T.reshape(y_test_T.shape[1], digits)
        y_test_T = y_test_T.T

        #  Normalizing data(RGB interval = 0 ~ 255)
        X_test_T = X_test_T / 255.

        return X_test_T, y_test_T

    def _preprocess_train_val(self, X_train, y_train, X_val, y_val):
        """
            Handle training and validation data for training.
        """
        # Transpose the data
        y_train_T = y_train.T 
        y_val_T = y_val.T

        X_train_T = X_train.T 
        X_val_T = X_val.T

        # One hot encoding
        digits = 10
        y_train_T = np.eye(digits)[y_train_T.astype('int32')]
        y_val_T = np.eye(digits)[y_val_T.astype('int32')]

        y_train_T = y_train_T.reshape(self.n_samples, digits)
        y_val_T = y_val_T.reshape(y_val_T.shape[1], digits)

        y_train_T = y_train_T.T
        y_val_T = y_val_T.T

        #  Normalizing data(RGB interval = 0 ~ 255)
        X_train_T = X_train_T / 255.
        X_val_T = X_val_T / 255.

        return X_train_T, y_train_T, X_val_T, y_val_T
