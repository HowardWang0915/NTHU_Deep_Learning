import numpy as np

def Linear(input_dim, output_dim):
    """
        Initalize a linear layer, like in pytorch.
        The method used is He initialization.
        :input_dim: input dimension
        :output_dim: output dimension
        :return: weights and biases
    """
    params = {'W': np.random.randn(output_dim, input_dim) * np.sqrt(1. / input_dim),
              'b': np.zeros((output_dim, 1))}
              #  'b': np.random.randn(output_dim, 1) * np.sqrt(1. /  input_dim)}
    return params

