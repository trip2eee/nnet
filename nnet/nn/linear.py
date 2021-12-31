from nnet.module import Module
import numpy as np
from nnet.optim.optimizer import Optimizer

class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.rnd_mean = 0.0
        self.rnd_std = 0.003

        weight = np.random.normal(self.rnd_mean, self.rnd_std, (in_features, out_features)).astype(np.float32)
        bias = np.zeros(out_features).astype(np.float32)

        self.pm = {'w':weight, 'b':bias}        
        

    def forward(self, x):
        y = np.matmul(x, self.pm['w']) + self.pm['b']        
        self.x = x
        self.y = y

        return y

    def backward(self, G_y, optim: Optimizer):
        x = self.x
        pm = self.pm

        g_y_weight = x.transpose()
        g_y_input = pm['w'].transpose()

        G_weight = np.matmul(g_y_weight, G_y)
        G_bias = np.sum(G_y, axis=0)
        G_x = np.matmul(G_y, g_y_input)
        
        optim.update_param(pm, 'w', G_weight)
        optim.update_param(pm, 'b', G_bias)

        return G_x

