"""
Reference: https://github.com/KONANtechnology/Academy.ALZZA/blob/master/codes/chap07/cnn_basic_model.ipynb
"""
from nnet.module import Module
import numpy as np
from nnet.optim.optimizer import Optimizer

class MaxPool2d(Module):
    def __init__(self, kernel_size):
        """
        Initializes 2D max pooling layer.        
        """
        super(MaxPool2d, self).__init__()
        
        self.kernel_size = kernel_size

    def forward(self, x):
        # (batch, input height, input width, input channels)
        batches, xh, xw, xc = x.shape    
        sh, sw = self.kernel_size

        yh = xh // sh
        yw = xw // sw
        
        x1 = x.reshape([batches, yh, sh, yw, sw, xc])
        # (batches, yh, yw, xc, sh, sw)
        x2 = x1.transpose(0, 1, 3, 5, 2, 4)
        x3 = x2.reshape([-1, sh*sw])

        idx_max = np.argmax(x3, axis=1)
        y_flat = x3[np.arange(batches * yh * yw * xc), idx_max]
        y = y_flat.reshape([batches, yh, yw, xc])

        self.idx_max = idx_max
        
        return y

    def backward(self, G_y, optim: Optimizer):
        batches, yh, yw, yc = G_y.shape
        idx_max = self.idx_max
        sh, sw = self.kernel_size

        xh = yh * sh
        xw = yw * sw

        gy_flat = G_y.flatten()
        gx1 = np.zeros([batches*yh*yw*yc, sh*sw], dtype=np.float32)

        gx1[np.arange(batches * yh * yw * yc), idx_max] = gy_flat[:]

        gx2 = gx1.reshape([batches, yh, yw, yc, sh, sw])
        gx3 = gx2.transpose([0, 1, 4, 2, 5, 3])
        G_x = gx3.reshape([batches, xh, xw, yc])

        return G_x

