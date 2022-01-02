from nnet.optim.optimizer import Optimizer

class Module:
    """
    Base module class
    """
    def __init__(self) -> None:
        self.is_training = True
        self.modules = []
        self.pm = {}

    def parameters(self):
        return self.modules        

    def forward(self, x):
        """
        Forward propagation
        """
        for module in self.modules:
            x = module.forward(x)
            
        return x

    def backward(self, G_y, optim : Optimizer):
        """
        Backward propagation
        """
        G_x = G_y

        for module in reversed(self.modules):
            G_x = module.backward(G_x, optim)

    def __call__(self, *x):
        return self.forward(*x)

    def train(self):
        """
        Set this module and its submodules to training mode.
        """
        self.is_training = True

        for module in self.modules:
            module.train()

    def eval(self):
        """
        Set this module and its submodules to evaluation mode.
        """
        self.is_training = False

        for module in self.modules:
            module.eval()