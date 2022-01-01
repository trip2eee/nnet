"""
Reference: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
"""

from typing import get_type_hints
import numpy as np
from nnet.optim.optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-8) -> None:
        super(Adam, self).__init__()

        self.parameters = parameters
        self.learning_rate = lr
        self.betas = betas
        self.eps = eps

    def update_param(self, pm, key, gt):
        gt = self.eval_adam_delta(pm, key, gt)

        pm[key] -= self.learning_rate * gt
    
    def eval_adam_delta(self, pm, key, gt):
        beta1 = self.betas[0]
        beta2 = self.betas[1]
        eps = self.eps
        
        key_mt, key_vt, key_t = 'm_'+key, 'v_'+key, 't_'+key
        if key_mt not in pm:
            pm[key_mt] = np.zeros(pm[key].shape)
            pm[key_vt] = np.zeros(pm[key].shape)
            pm[key_t] = 0
        
        mt = beta1 * pm[key_mt] + (1 - beta1) * gt
        vt = beta2 * pm[key_vt] + (1 - beta2) * (gt * gt)

        pm[key_mt] = mt
        pm[key_vt] = vt

        # t > 0. Otherwise, mt and vt are divided by zero.
        pm[key_t] += 1        
        t = pm[key_t]

        mt = mt / (1 - (beta1 ** t))
        vt = vt / (1 - (beta2 ** t))

        return mt / (np.sqrt(vt) + eps)





