import numpy as np 

class BaseDataLoader():
    """ 
        Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, val_split, n_workers):
        self.val_split = val_split
        self.shuffle = shuffle 
        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.dataset = dataset
        if self.val_split != 0.0:
            self.X_train, self.y_train, self.X_val, self.y_val = self._train_val_split(self.val_split)
        self.batch_size = batch_size
        self.n_workers = n_workers
        
    def _train_val_split(self, split):
        """ 
            Perform train val split on the data
            :split: the ratio to split your training and validation data.
        """

        # Don't split if split is set to 0.0
        if split == 0.0:
            return self.dataset.X_train, self.dataset.y_train, None, None
        idx_full = np.arange(self.n_samples)

        # start generating shuffled test/valid split
        np.random.seed(0)
        np.random.shuffle(idx_full)

        # Calculate the size for validation
        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is wrong!"
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        # Retreive the validation and training indices
        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        X_train = self.dataset.X_train[train_idx]
        y_train = self.dataset.y_train[train_idx]
        X_val = self.dataset.X_train[valid_idx]
        y_val = self.dataset.y_train[valid_idx]
        self.n_samples = len(train_idx)

        return X_train, y_train, X_val, y_val

    def _shuffle(self, X, y):
        """
            Shuffle input data
        """
        shuffle_idx = np.random.permutation(self.n_samples)
        shuffle_x, shuffle_y = X[:, shuffle_idx], y[:, shuffle_idx]
        return shuffle_x, shuffle_y


    def __len__(self):
        """
            A len method to show the size of dataset
        """
        return int(self.n_samples / self.batch_size)

    def __getitem__(self, key):
        """
            Return mini-batches for your data
        """
        # Use the data loader as key, prevent data modification
        if key == 0:
            self.X_train, self.y_train = self._shuffle(self.X_train, self.y_train)
        if key < int(self.n_samples / self.batch_size) + 1:
            begin = key * self.batch_size
            end = min(begin + self.batch_size, self.X_train.shape[1] - 1)
            X = self.X_train[:, begin:end]
            y = self.y_train[:, begin:end]
            return X, y
        else:
            raise IndexError

