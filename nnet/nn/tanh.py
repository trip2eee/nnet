from nnet.module import Module
from nnet.optim.optimizer import Optimizer
import nnet.nn.math as math

class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.pm = None

    def forward(self, x):
        y = math.tanh(x)
        self.y = y
        
        return y

    def backward(self, G_y, optim: Optimizer):
        G_x = math.tanh_derv(self.y) * G_y

        return G_x

