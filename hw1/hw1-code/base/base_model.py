from abc import abstractmethod
import numpy as np

class BaseModel():
    """ 
        Base class for all models
    """
    def __init__(self, input_dim, output_dim, n_weights):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_weights = n_weights
        # Remember all calculated tensors
        self.tensors = dict()
        pass

    @abstractmethod
    def forward(self, *inputs):
        """
            Forward pass logic
        """
        raise NotImplementedError

    # This is useful for showing model details
    def __str__(self):
        """
            Model prints with number of trainable parameters
        """
        return super().__str__() + '\nTrainable params: {}'.format(self.n_weights)

