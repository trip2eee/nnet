import numpy as np

def relu(x):
    return np.maximum(x, 0)

def relu_derv(x):
    # y = relu(x) is not differentiable at x = 0.
    # if y > 0, derivative = 1
    # otherwise, derivative = 0
    return np.sign(x)
    
def sigmoid(x):
    # s(x) = 1 / (1 + exp(-x))

    return np.exp(-np.maximum(-x, 0)) / (1.0 + np.exp(-np.abs(x)))

def sigmoid_derv(x):
    # derivative of sigmoid.
    # s(x) = 1 / (1 + exp(-x))
    # ds/dx = -(1+exp(-x))' / (1+exp(-x))^2
    # = exp(-x) / (1+exp(-x))^2
    # = ((1+exp(-x)) - 1) / (1 + exp(-x))^2
    # = 1/(1+exp(-x)) * (1 - 1/exp(-x))
    # = s(x)  * (1 - s(x))
    return x * (1 - x)
    
def tanh(x):
    return 2 * sigmoid(2*x) - 1

def tanh_derv(x):
    return (1.0 + x) * (1.0 - x)


def sigmoid_cross_entropy_with_logits(pt, x):
    # pt: p(true)
    # 1-pt : p(false)
    return np.maximum(x, 0) - x * pt + np.log(1 + np.exp(-np.abs(x)))

def sigmoid_cross_entropy_with_logits_derv(pt, x):
    # pt: p(true)
    # 1-pt : p(false)
    return -pt + sigmoid(x)

def softmax(x):
    # x: (batch, class)
    max_elem = np.max(x, axis=1)
    diff = (x.transpose() - max_elem).transpose()
    exp = np.exp(diff)
    sum_exp = np.sum(exp, axis=1)
    probs = (exp.transpose() / sum_exp).transpose()
    return probs

def softmax_cross_entropy_with_logits(labels, logits):
    # H(P,Q) = -SUM p_i log (q_i + eps)
    # eps: epsilon
    eps = 1.0e-10

    probs = softmax(logits)
    return -np.sum(labels * np.log(probs + eps), axis=1)

def softmax_cross_entropy_with_logits_derv(labels, logits):

    return softmax(logits) - labels

def onehot(x, dim_x):
    y = np.eye(dim_x)[np.array(x).astype(int)]
    return y
