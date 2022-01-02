from nnet.module import Module
import numpy as np
from nnet.optim.optimizer import Optimizer

class BatchNorm2d(Module):
    """
    Batch normalization in 2D over channel C of input (N, H, W, C).
    y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
    """
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        """
        nm_features (int) Number of input channels.
        momentum (float) The value used for the nunning mean and running var computation. default: 0.1
        eps (float) A value added to the denominator for numerical stability. default: 1e-5.
        """
        super(BatchNorm2d, self).__init__()
        self.momentum = momentum
        self.eps = eps
        self.num_features = num_features
        
        # input shape is not defined yet.
        self.pm = {}
        self.pm['mavg'] = np.zeros(num_features)     # moving average
        self.pm['mvar'] = np.zeros(num_features)     # moving variance
        self.pm['gamma'] = np.ones(num_features)     # scale
        self.pm['beta'] = np.zeros(num_features)     # bias

    def forward(self, x):
        if self.is_training:
            x_flat = x.reshape((-1, self.num_features))
            avg = np.mean(x_flat, axis=0)
            var = np.var(x_flat, axis=0)

            momentum = self.momentum
            self.pm['mavg'] += momentum * (avg - self.pm['mavg'])
            self.pm['mvar'] += momentum * (var - self.pm['mvar'])
        else:
            avg = self.pm['mavg']
            var = self.pm['mvar']

        std = np.sqrt(var + self.eps)
        x_norm = (x - avg) / std

        self.x_norm = x_norm
        self.std = std

        y = self.pm['gamma'] * x_norm + self.pm['beta']

        return y

    def backward(self, G_y, optim: Optimizer):
        x_norm = self.x_norm
        std = self.std

        # for 2D input, summation over batch, height, width.
        G_gamma = np.sum(G_y*x_norm, axis=(0, 1, 2))
        G_beta  = np.sum(G_y, axis=(0, 1, 2))

        G_y = G_y * self.pm['gamma'] # scale
        G_x = G_y / std
        
        self.pm['gamma'] -= optim.learning_rate * G_gamma
        self.pm['beta'] -= optim.learning_rate * G_beta

        return G_x



