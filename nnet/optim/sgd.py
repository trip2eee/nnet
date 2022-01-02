"""
Reference: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
"""

import numpy as np
from nnet.optim.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr, momentum=0, dampening=0, weight_decay_l1=0.0, weight_decay_l2=0.0):
        super(SGD, self).__init__(parameters, lr, weight_decay_l1=weight_decay_l1, weight_decay_l2=weight_decay_l2)
        self.momentum = momentum
        self.dampening = dampening
    
    def update_param(self, pm, key, gt):
        key_t = 't_' + key
        key_b = 'b_' + key
        mu = self.momentum
        tau = self.dampening
        
        if mu > 0:
            if key_t not in pm:
                pm[key_t] = 0
                pm[key_b] = np.zeros(gt.shape)
    
            t = pm[key_t]

            if t > 0:
                bt = mu*pm[key_b] + (1-tau)*gt
            else:
                bt = gt

            pm[key_t] += 1
            pm[key_b] = bt

            gt = bt
        
        gt = self.regularize(pm, key, gt)
        pm[key] -= self.learning_rate * gt
        


