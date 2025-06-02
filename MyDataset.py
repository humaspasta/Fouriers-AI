import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data #scalars representing time

        self.labels = labels # tensor representing positions (x,y) that the point should be at

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx] , dtype=torch.float32)
        y = torch.tensor(self.labels[idx] , dtype=torch.float32)
        return x, y