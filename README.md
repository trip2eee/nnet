# NNet
NNet is a neural network library written for study.

* This library mimics PyTorch interfaces and packages.
* This library only supports CPU computations.

# MNIST Example
## Dataset download
Download MNIST dataset from the following site.

http://yann.lecun.com/exdb/mnist/

## Dataset class

mnist_dataset.py

```python
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
```

### Training
Training is performed only for 10 epochs because this library supports only CPU computations. The training takes several minutes.

train_mnist.py
```python
import nnet
import nnet.nn as nn
import numpy as np
from mnist_dataset import MNISTDataset

NUM_CLASSES = 10
class ModelMNIST(nnet.Model):
    def __init__(self):
        super(ModelMNIST, self).__init__()

        self.layers = [
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Flatten(),
            nn.Linear(in_features=7*7*8, out_features=NUM_CLASSES),
        ]

def accuracy(output, target):
    estimate = np.argmax(output, axis=1)
    answer = np.argmax(target, axis=1)
    correct = np.equal(estimate, answer)

    return np.mean(correct)
    
if __name__ == "__main__":
    train_set = MNISTDataset('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte')
    train_data = nnet.DataLoader(train_set, batch_size=100, shuffle=True)

    test_set = MNISTDataset('mnist/t10k-images.idx3-ubyte', 'mnist/t10k-labels.idx1-ubyte')
    test_data = nnet.DataLoader(test_set, batch_size=-1, shuffle=False)

    model = ModelMNIST()
    celoss = nnet.loss.CELoss()
    optim = nnet.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(10):
        losses = []
        accs = []
        for idx_batch, (x, y) in enumerate(train_data):
            output = model(x)
            loss = celoss(output, y)
            optim.step(celoss.backward(1.0))
            losses.append(loss)

            acc = accuracy(output, y)
            accs.append(acc)
            
        print('Epoch {}: loss={:0.3f}, accuracy={:0.3f}'.format(epoch+1, np.mean(losses), np.mean(accs)))

    accs = []
    for idx_batch, (x, y) in enumerate(test_data):
        output = model(x)
        acc = accuracy(output, y)
        accs.append(acc)
            
    print('Test accuracy={:0.3f}'.format(np.mean(accs)))
```

### Training result
The result of training for 10 epochs.
```
Epoch 1: loss=1.715, accuracy=0.519
Epoch 2: loss=0.656, accuracy=0.810
Epoch 3: loss=0.506, accuracy=0.848
Epoch 4: loss=0.452, accuracy=0.864
Epoch 5: loss=0.423, accuracy=0.874
Epoch 6: loss=0.403, accuracy=0.880
Epoch 7: loss=0.388, accuracy=0.885
Epoch 8: loss=0.377, accuracy=0.888
Epoch 9: loss=0.368, accuracy=0.892
Epoch 10: loss=0.361, accuracy=0.895
Test accuracy=0.897
```

## References
https://github.com/KONANtechnology/Academy.ALZZA

