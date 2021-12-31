import csv
import time
import numpy as np
from nnet.loss.loss import Loss
from nnet.optim.optimizer import Optimizer

class Model:
    def __init__(self, dim_input, dim_output, hidden_config, rnd_mean=0, rnd_std=0.0030):

        self.is_training = False

        self.dim_input = dim_input
        self.dim_output = dim_output
        
        self.rnd_mean = rnd_mean
        self.rnd_std = rnd_std
        self.hidden_config = hidden_config
        
        np.random.seed(123)

    def randomize(self):
        np.random.seed(time.time())

    def init_model_hiddens(self):
        self.pm_hiddens = []
        prev_dim = self.dim_input

        for hidden_dim in self.hidden_config:
            self.pm_hiddens.append(self.alloc_param_pair([prev_dim, hidden_dim]))
            prev_dim = hidden_dim
        
        self.pm_output = self.alloc_param_pair([prev_dim, self.dim_output])
        

    def alloc_param_pair(self, shape):
        weight = np.random.normal(self.rnd_mean, self.rnd_std, shape).astype(np.float32)
        bias = np.zeros(shape[-1]).astype(np.float32)
        return {'w':weight, 'b':bias}

    def train(self, x, y, loss_obj : Loss, optim : Optimizer):
        output, aux_nn = self.forward(x)
        loss, aux_pp = loss_obj.forward(output, y)

        G_loss = 1.0
        G_output = loss_obj.backward(G_loss, aux_pp)
        self.backward(G_output, aux_nn, optim)

        return output, loss

    def test(self, x, y):
        output, _ = self.forward(x)
        
        return output
    
    def relu(self, x):
        return np.maximum(x, 0)

    
    def relu_derv(self, y):
        # y = relu(x) is not differentiable at x = 0.
        # if y > 0, derivative = 1
        # otherwise, derivative = 0
        return np.sign(y)

    def forward(self, x):

        hidden = x
        aux_layers = []
        hiddens = [x]

        for pm_hidden in self.pm_hiddens:
            hidden, aux = self.forward_layer(hidden, pm_hidden, 'relu')
            aux_layers.append(aux)

        output, aux_out = self.forward_layer(hidden, self.pm_output, None)
        
        return output, [aux_out, aux_layers]

    def forward_layer(self, x, pm, activation):
        y = np.matmul(x, pm['w']) + pm['b']
        if activation == 'relu':
            y = self.relu(y)
        return y, [x, y]
        

    def backward(self, G_output, aux, optim : Optimizer):
        aux_out, aux_layers = aux

        G_hidden = optim.step(G_output, None, self.pm_output, aux_out)

        for n in reversed(range(len(self.pm_hiddens))):
            G_hidden = optim.step(G_hidden, 'relu', self.pm_hiddens[n], aux_layers[n])

    def accuracy(self, output, target):
        estimate = np.argmax(output, axis=1)
        answer = np.argmax(target, axis=1)
        correct = np.equal(estimate, answer)

        return np.mean(correct)
