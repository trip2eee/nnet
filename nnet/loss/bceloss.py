from nnet.loss.loss import Loss
import numpy as np

class BCELoss(Loss):
    def __init__(self):
        super(Loss, self).__init__()
    
    def forward(self, output, target):
        entropy = self.sigmoid_cross_entropy_with_logits(target, output)
        loss = np.mean(entropy)

        self.target = target
        self.output = output
        self.entropy = entropy

        return loss

    def backward(self, G_loss):
        target = self.target
        output = self.output
        entropy = self.entropy

        g_loss_entropy = 1.0 / np.prod(entropy.shape)
        g_entropy_output = self.sigmoid_cross_entropy_with_logits_derv(target, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy

        return G_output

    def sigmoid(self, x):
        # s(x) = 1 / (1 + exp(-x))

        return np.exp(-np.maximum(-x, 0)) / (1.0 + np.exp(-np.abs(x)))

    def sigmoid_derv(self, x, y):
        # derivative of sigmoid.
        # s(x) = 1 / (1 + exp(-x))
        # ds/dx = -(1+exp(-x))' / (1+exp(-x))^2
        # = exp(-x) / (1+exp(-x))^2
        # = ((1+exp(-x)) - 1) / (1 + exp(-x))^2
        # = 1/(1+exp(-x)) * (1 - 1/exp(-x))
        # = s(x)  * (1 - s(x))
        return y * (1 - y)

    def sigmoid_cross_entropy_with_logits(self, pt, x):
        # pt: p(true)
        # 1-pt : p(false)
        return np.maximum(x, 0) - x * pt + np.log(1 + np.exp(-np.abs(x)))

    def sigmoid_cross_entropy_with_logits_derv(self, pt, x):
        # pt: p(true)
        # 1-pt : p(false)
        return -pt + self.sigmoid(x)