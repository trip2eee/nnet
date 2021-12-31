class Module:
    def __init__(self) -> None:
        pass

    def forward(self, *x) -> None:
        pass

    def backward(self, *x) -> None:
        pass

    def __call__(self, *x):
        return self.forward(*x)
