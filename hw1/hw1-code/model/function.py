import numpy as np 

""" 
    This file contains all custom functions to be used
"""

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    """
        Compute softmax of x
    """
    exps = np.exp(x)
    return exps / np.sum(exps, axis=0)
