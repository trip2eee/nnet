from nnet.module import Module
from nnet.optim.optimizer import Optimizer
import nnet.nn.math as math

class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        y = math.relu(x)
        self.y = y
        
        return y

    def backward(self, G_y, optim: Optimizer):
        G_x = math.relu_derv(self.y) * G_y

        return G_x

