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

    @abstractmethod
    def forward(self, *inputs):
        """
            Forward pass logic
        """
        raise NotImplementedError
    @abstractmethod
    def backward(self, *inputs):
        """
            Backward pass logic
        """
        raise NotImplementedError

    @abstractmethod
    def get_layer_weights(self):
        """
            Return all the layers that contains weights
        """
        raise NotImplementedError

    # This is useful for showing model details
    def __str__(self):
        """
            Model prints with number of trainable parameters
        """
        return super().__str__() + '\nTrainable params: {}'.format(self.n_weights)

