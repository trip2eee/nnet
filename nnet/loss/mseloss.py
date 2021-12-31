from nnet.loss.loss import Loss
import numpy as np

class MSELoss(Loss):
    def __init__(self):
        super(Loss, self).__init__()
    
    def forward(self, output, target):
        # Mean Squared Error (MSE)
        # diff -> square -> mean
        diff = output - target
        square = np.square(diff)
        loss = np.mean(square)

        self.diff = diff

        return loss

    def backward(self, G_loss):
        # square -> diff -> output
        diff = self.diff
        shape = diff.shape

        # dL/dSQR(i,j) = 1/MN
        g_loss_sqr = np.ones(shape) / np.prod(shape)

        # SQR(i,j) = DIFF(i,j)^2
        # dSQR(i,j)/dDIFF(i,j) = 2*DIFF(i,j)
        g_sqr_diff = 2*diff

        # DIFF(i,j) = OUTPUT(i,j) - y(i,j)
        # dDIFF(i,j)/dOUTPUT(i,j) = 1
        g_diff_output = 1

        # dDIFF(i,j)/dOUPUT(i,j) * dSQR(i,j)/dDIFF(i,j) * dL/dSQR(i,j) * dG/dL = dG/dOUTPUT(i,j)
        G_sqr = g_loss_sqr * G_loss
        G_diff = g_sqr_diff * G_sqr
        G_output = g_diff_output * G_diff

        return G_output

