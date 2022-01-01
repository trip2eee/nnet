import numpy as np
import os
from nnet.dataset import Dataset
import nnet.nn.math

NUM_CLASSES = 10
class MNISTDataset(Dataset):    
    def __init__(self, path):
        super(MNISTDataset, self).__init__()
        images = np.fromfile(os.path.join(path, 'train-images.idx3-ubyte'), dtype=np.uint8)        
        magic_number = self.byte2int(images[0:4])
        assert(magic_number == 2051)
        num_samples = self.byte2int(images[4:8])
        assert(num_samples == 60000)

        labels = np.fromfile(os.path.join(path, 'train-labels.idx1-ubyte'), dtype=np.uint8)
        magic_number = self.byte2int(labels[0:4])
        assert(magic_number == 2049)
        num_samples = self.byte2int(labels[4:8])
        assert(num_samples == 60000)

        rows = self.byte2int(images[8:12])
        cols = self.byte2int(images[12:16])

        self.x = np.reshape(images[16:], (num_samples, rows, cols, 1)).astype(np.float32) / 255.0        
        self.y = nnet.nn.math.onehot(labels[8:], NUM_CLASSES)   # one hot encoding
    
    def byte2int(self, bytes):
        val = 0
        for b in bytes:
            val = (val << 8) + int(b)
        return val

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):        
        x = self.x[idx]
        y = self.y[idx]
        return x, y


