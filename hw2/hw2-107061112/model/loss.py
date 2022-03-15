import torch.nn as nn

def MSELoss(output, target):
    return nn.MSELoss()(output, target)
