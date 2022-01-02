import numpy as np

class Optimizer:
    def __init__(self, parameters, lr, weight_decay_l1=0.0, weight_decay_l2=0.0):
        self.parameters = parameters
        self.learning_rate = lr
        self.weight_decay_l1 = weight_decay_l1
        self.weight_decay_l2 = weight_decay_l2

    def step(self, G_y):
        """
        Backward propagation
        """        
        if self.weight_decay_l1 > 0.0 or self.weight_decay_l2 > 0.0:
            params = self.collect_params()
            params = np.asarray(params)
            
            self.num_params = len(params)

            # L1 regularization
            if self.weight_decay_l1 > 0.0:
                G_y += self.weight_decay_l1 * np.sum(np.abs(params)) / self.num_params
            
            # L2 regularization
            if self.weight_decay_l2 > 0.0:
                G_y += self.weight_decay_l2 * np.sum(np.square(params)) / (2.0 * self.num_params)
            
        G_x = G_y
        for param in reversed(self.parameters):
            G_x = param.backward(G_x, self)

    def regularize(self, pm, key, gt):
        if key == 'w':
            if self.weight_decay_l1 > 0.0:
                gt += self.weight_decay_l1 * np.sign(pm[key]) / self.num_params
            if self.weight_decay_l2 > 0.0:
                gt += self.weight_decay_l2 * pm[key] / self.num_params
        
        return gt

    def update_param(self, pm, key, gt) -> None:
        raise NotImplementedError

    def collect_params(self):
        params = []
        for param in self.parameters:
            if 'w' in param.pm:
                w = param.pm['w']
                params += list(w.flatten())
        return params