import numpy as np

"""
    This file will define all custom losses in this project
    Each loss class should contain forward and backward passes
"""

class cross_entropy():
    """
        Implement cross-entropy loss
        output: The output prediction of our model
        target: The actual label of our data.
    """
    def __init__(self):
        pass

    def forward(self, output, target):
        """
            The foward pass of the cross_entropy loss
        """
        self.y_hat = output
        self.y = target
        loss = np.sum(np.multiply(self.y_hat, np.log(self.y + 1e-8)))
        m = target.shape[1]
        loss = -(1. / m) * loss
        return loss

    def backward(self, output, target):
        """
            The backward pass of the cross_entropy + softmax combined
        """
        return output - target

