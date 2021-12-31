from nnet.loss.loss import Loss
import numpy as np

class CELoss(Loss):
    def __init__(self):
        super(Loss, self).__init__()
    
    def forward(self, output, target):
        entropy = self.softmax_cross_entropy_with_logits(target, output)
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
        g_entropy_output = self.softmax_cross_entropy_with_logits_derv(target, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy

        return G_output

    def softmax(self, x):
        # x: (batch, class)
        max_elem = np.max(x, axis=1)
        diff = (x.transpose() - max_elem).transpose()
        exp = np.exp(diff)
        sum_exp = np.sum(exp, axis=1)
        probs = (exp.transpose() / sum_exp).transpose()
        return probs

    def softmax_cross_entropy_with_logits(self, labels, logits):
        # H(P,Q) = -SUM p_i log (q_i + eps)
        # eps: epsilon
        eps = 1.0e-10

        probs = self.softmax(logits)
        return -np.sum(labels * np.log(probs + eps), axis=1)

    def softmax_cross_entropy_with_logits_derv(self, labels, logits):

        return self.softmax(logits) - labels