import nnet
from nnet.loss.celoss import CELoss
import nnet.nn as nn
import nnet.loss
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

    train_set = MNISTDataset('mnist')
    train_data = nnet.DataLoader(train_set, batch_size=100, shuffle=True)
    
    model = ModelMNIST()
    celoss = CELoss()

    optim = nnet.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(10):
        losses = []
        accs = []

        for idx_batch, (x, y) in enumerate(train_data):
            output = model(x)
            loss = celoss(output, y)

            optim.step(celoss.backward(1.0))

            acc = accuracy(output, y)

            losses.append(loss)
            accs.append(acc)
            
        print('Epoch {}: loss={:5.4f}, accuracy={:5.3f}'.format(epoch+1, np.mean(losses), np.mean(accs)))


