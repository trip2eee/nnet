from nnet.module import Module

class Optimizer(Module):
    def __init__(self):
        super(Optimizer, self).__init__()
        self.parameters = []

    def step(self, G_y):
        """
        Backward propagation
        """
        G_x = G_y
        for param in reversed(self.parameters):
            G_x = param.backward(G_x, self)
        

    def update_param(self, pm, key, gt) -> None:
        pass