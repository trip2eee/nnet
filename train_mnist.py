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


