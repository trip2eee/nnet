import numpy as np
from numpy.random.mtrand import shuffle

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset        
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

        if batch_size == -1:
            self.batch_size = len(dataset)
        else:
            self.batch_size = batch_size

        self.idx_iter = 0
        np.random.seed(123)


    def __len__(self):        
        num_data = len(self.dataset)
        num_batches = np.ceil(num_data / self.batch_size)

        return int(num_batches)

    def __iter__(self):
        self.idx_iter = 0
        np.random.shuffle(self.indices)

        return self        

    def __next__(self):
        idx_iter = self.idx_iter
        if idx_iter >= len(self):
            raise StopIteration

        self.idx_iter += 1

        idx0 = idx_iter * self.batch_size
        idx1 = min(len(self.dataset), (idx_iter + 1) * self.batch_size)

        x, y = self.dataset[self.indices[idx0:idx1]]

        return x, y
        
    def __getitem__(self, idx_batch):
        if idx_batch == 0 and shuffle:
            np.random.shuffle(self.indices)
        
        if idx_batch >= len(self):
            raise IndexError

        idx0 = idx_batch * self.batch_size
        idx1 = min(len(self.dataset), (idx_batch + 1) * self.batch_size)

        x, y = self.dataset[self.indices[idx0:idx1]]
        
        return x, y



        