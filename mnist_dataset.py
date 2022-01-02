import numpy as np
from nnet.dataset import Dataset
import nnet.nn.math

NUM_CLASSES = 10
class MNISTDataset(Dataset):
    def __init__(self, image_path, label_path):
        super(MNISTDataset, self).__init__()
        images = np.fromfile(image_path, dtype=np.uint8)        
        labels = np.fromfile(label_path, dtype=np.uint8)
        
        # first 32-bit: magic number, second 32-bit: number of items
        rows = self.byte2int(images[8:12])
        cols = self.byte2int(images[12:16])
        self.x = np.reshape(images[16:], (-1, rows, cols, 1)).astype(np.float32) / 255.0

        # one hot encoding of label. first 32-bit: magic number, second 32-bit: number of items
        self.y = nnet.nn.math.onehot(labels[8:], NUM_CLASSES)
    
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


