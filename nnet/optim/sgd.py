"""
Reference: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
"""

import numpy as np
from nnet.optim.optimizer import Optimizer
import nnet.math as math

class SGD(Optimizer):
    def __init__(self, parameters, lr, momentum=0, dampening=0) -> None:
        super(SGD, self).__init__()
        
        self.parameters = parameters
        self.learning_rate = lr
        self.momentum = momentum
        self.dampening = dampening
    
    def update_param(self, pm, key, gt):
        key_t = 't_' + key
        key_b = 'b_' + key
        mu = self.momentum
        gamma = self.dampening
        
        if mu > 0:
            if key_t not in pm:
                pm[key_t] = 0
                pm[key_b] = np.zeros(gt.shape)
    
            t = pm[key_t]

            if t > 0:
                bt = mu*pm[key_b] + (1-gamma)*gt
            else:
                bt = gt

            pm[key_t] += 1
            pm[key_b] = bt

            gt = bt
            
        pm[key] -= self.learning_rate * gt
        


