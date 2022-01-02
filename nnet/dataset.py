import numpy as np

class Dataset:
    def __init__(self)->None:
        pass

    def __len__(self):        
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

