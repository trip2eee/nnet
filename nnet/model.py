import csv
import time
import numpy as np
from nnet.loss.loss import Loss
from nnet.optim.optimizer import Optimizer
from nnet.module import Module

class Model(Module):
    def __init__(self):
        """
        Initializes internal state.
        """
        self.is_training = False
        self.layers = []

        np.random.seed(123)

    def randomize(self):
        np.random.seed(time.time())

    def parameters(self):
        return self.layers        

    def train(self, x, y, loss_obj : Loss, optim : Optimizer):
        output = self.forward(x)
        loss = loss_obj.forward(output, y)

        G_loss = 1.0
        G_output = loss_obj.backward(G_loss)
        self.backward(G_output, optim)

        return output, loss

    def forward(self, x):
        """
        Forward propagation
        """
        for layer in self.layers:
            x = layer.forward(x)
            
        return x


    def backward(self, G_y, optim : Optimizer):
        """
        Backward propagation
        """
        G_x = G_y

        for layer in reversed(self.layers):
            G_x = layer.backward(G_x, optim)
