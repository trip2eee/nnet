from nnet.module import Module
import numpy as np
from nnet.optim.optimizer import Optimizer

class Dropout(Module):
    """
    During training, randomly zeroes some of the elements of the input tensor with probability of p.
    """
    def __init__(self, p=0.5):
        """
        p (float) probability of an element to be zeroed. default: 0.5
        """
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.is_training:
            # number of trial: 1
            mask = np.random.binomial(1, 1.0-self.p, x.shape)
            y = x * mask / (1 - self.p)
            self.mask = mask
        else:
            y = x
        return y

    def backward(self, G_y, optim: Optimizer):
        # y = (x * mask) / (1 - p)
        # dy/dx = mask/(1-p)
        G_x = G_y * self.mask * (1 - self.p)

        return G_x



