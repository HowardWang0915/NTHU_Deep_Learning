from abc import abstractmethod

class BaseDataset():
    """
        A base dataset class, Inherit this when using your own dataset
    """
    def __init__(self, data_dir, train):
        # location of the dataset
        self.data_dir = data_dir
        # Use training data? or test data?
        self.train = train

        self.X_train, self.y_train = self._extract_training_data(self.data_dir)
        self.X_test, self.y_test = self._extract_test_data(self.data_dir)

    @abstractmethod
    def _extract_training_data(self, data_dir):
        """
            logic for extracting training data
        """
        raise NotImplementedError

    @abstractmethod
    def _extract_test_data(self, data_dir):
        """
            logic for extracting testing data
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.y_train)

