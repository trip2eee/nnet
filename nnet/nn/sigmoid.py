from nnet.module import Module
from nnet.optim.optimizer import Optimizer
import nnet.math as math

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.pm = None

    def forward(self, x):
        y = math.sigmoid(x)
        self.y = y
        
        return y

    def backward(self, G_y, optim: Optimizer):
        G_x = math.sigmoid_derv(self.y) * G_y

        return G_x

