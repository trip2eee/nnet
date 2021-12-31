"""
Reference: https://github.com/KONANtechnology/Academy.ALZZA/blob/master/codes/chap07/cnn_basic_model.ipynb
"""
from nnet.module import Module
import numpy as np
from nnet.optim.optimizer import Optimizer

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        """
        Initializes 2D convolution layer.
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (tuple) - Size of the convolving kernel
        stride
        """
        super(Conv2d, self).__init__()
        self.rnd_mean = 0.0
        self.rnd_std = 0.003

        self.in_channels = in_channels
        self.out_channels = out_channels

        weight = np.random.normal(self.rnd_mean, self.rnd_std, (kernel_size[0], kernel_size[1], in_channels, out_channels)).astype(np.float32)
        bias = np.zeros(out_channels).astype(np.float32)

        self.pm = {'w':weight, 'b':bias}        
        
    def get_ext_regions(self, x, kh, kw, fill):
        # TODO: To remove fill
        batches, xh, xw, xc = x.shape

        eh = xh + kh - 1
        ew = xw + kw - 1

        bh = (kh-1) // 2
        bw = (kw-1) // 2

        # padded input.
        x_pad = np.zeros((batches, eh, ew, xc), dtype = np.float32) + fill
        x_pad[:, bh:bh+xh, bw:bw+xw, :] = x

        regs = np.zeros((xh, xw, batches*kh*kw*xc), dtype = np.float32)

        for r in range(xh):
            for c in range(xw):
                regs[r, c, :] = x_pad[:, r:r+kh, c:c+kw,:].flatten()

        return regs.reshape([xh, xw, batches, kh, kw, xc])

    def get_ext_regions_for_conv(self, x, kh, kw):
        batches, xh, xw, xc = x.shape

        regs = self.get_ext_regions(x, kh, kw, 0)

        # (batch, height, width, kh, kw, xc)
        regs = regs.transpose([2, 0, 1, 3, 4, 5])
        
        return regs.reshape([batches * xh * xw, kh*kw*xc])

    def undo_ext_regions(self, regs, kh, kw):
        xh, xw, batches, kh, kw, xc = regs.shape

        eh = xh + kh - 1
        ew = xw + kw - 1

        bh = (kh-1)//2
        bw = (kw-1)//2

        gx_ext = np.zeros([batches, eh, ew, xc], dtype=np.float32)

        for r in range(xh):
            for c in range(xw):
                gx_ext[:, r:r+kh, c:c+kw, :] += regs[r, c]
            
        return gx_ext[:, bh:bh+xh, bw:bw+xw, :]

    def undo_ext_regions_for_conv(self, regs, x, kh, kw):
        batches, xh, xw, xc = x.shape

        regs = regs.reshape([batches, xh, xw, kh, kw, xc])
        regs = regs.transpose([1, 2, 0, 3, 4, 5])

        return self.undo_ext_regions(regs, kh, kw)

    def forward(self, x):

        # (batch, input height, input width, input channels)
        batches, xh, xw, xc = x.shape

        # (kernel height, kernel width, input channels, output channels)
        kh, kw, _, yc = self.pm['w'].shape

        # (batches * xh * xw, kh*kw*xc)
        x_flat = self.get_ext_regions_for_conv(x, kh, kw)
        # (kh * kw * xc, yc)
        k_flat = self.pm['w'].reshape([kh*kw*xc, yc])

        # (batches * xh * xw, yc)
        y_flat = np.matmul(x_flat, k_flat)

        #( batches, xh, xw, yc)
        y = y_flat.reshape([batches, xh, xw, yc])

        # add bias
        y = y + self.pm['b']

        self.x = x
        self.y = y
        self.x_flat = x_flat
        self.k_flat = k_flat

        return y

    def backward(self, G_y, optim: Optimizer):
        x = self.x
        y = self.y
        x_flat = self.x_flat
        k_flat = self.k_flat

        pm = self.pm
        kh, kw, xc, yc = self.pm['w'].shape

        batches, yh, yw, _ = G_y.shape

        G_y_flat = G_y.reshape(batches * yh * yw, yc)

        g_y_k_flat = x_flat.transpose()
        g_y_x_flat = k_flat.transpose()

        G_k_flat = np.matmul(g_y_k_flat, G_y_flat)
        G_x_flat = np.matmul(G_y_flat, g_y_x_flat)
        G_bias = np.sum(G_y_flat, axis = 0)

        G_kernel = G_k_flat.reshape([kh, kw, xc, yc])
        G_x = self.undo_ext_regions_for_conv(G_x_flat, x, kh, kw)
        
        optim.update_param(pm, 'w', G_kernel)
        optim.update_param(pm, 'b', G_bias)

        return G_x



