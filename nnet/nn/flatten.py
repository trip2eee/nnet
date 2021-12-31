from nnet.module import Module
import numpy as np
from nnet.optim.optimizer import Optimizer

class Flatten(Module):
    """
    Flatten layer
    This layer flattens N-dimensional tensor into 1D array.
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batches, xh, xw, xc = x.shape
        y = x.reshape([batches, xh*xw*xc])
        self.x = x

        return y

    def backward(self, G_y, optim: Optimizer):
        """
        Input 1D array is converted into N-dimensional tensor (same as input x).
        """
        x = self.x
        batches, xh, xw, xc = x.shape

        G_x = G_y.reshape([batches, xh, xw, xc])

        return G_x



