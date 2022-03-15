import numpy as np

"""
    This file contains all custom metrics for evaluating your model
"""
def accuracy(output, target):
    """
        A simmple function to calculate the accuracy
    """
        
    pred = np.argmax(output, axis=0)
    y = np.argmax(target, axis=0)
    assert pred.shape == y.shape
    correct = 0
    correct += np.sum(pred == y)
    return correct / pred.shape[0]

