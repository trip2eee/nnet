class Optimizer:
    def __init__(self) -> None:
        self.parameters = []
        self.learning_rate = 0

    def step(self, G_y):
        """
        Backward propagation
        """
        G_x = G_y
        for param in reversed(self.parameters):
            G_x = param.backward(G_x, self)

    def update_param(self, pm, key, gt) -> None:
        raise NotImplementedError