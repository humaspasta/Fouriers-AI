import torch
from torch.utils.data import Dataset
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, time, x_points, y_points):
        '''
        Data: The actual output from the model
        Lables: The expected output
        '''
        if len(time) != len(x_points) or len(time) != len(y_points):
            raise ValueError("dataframe sizes are not the same")
        
        self.time = time #scalars representing overall rotation angle
        self.x = x_points # tensor representing position x given theta
        self.y = y_points #tensor representing position y given theta

    def __len__(self): 
        return len(self.time)

    def __getitem__(self, idx):
        return (self.time[idx] , self.x[idx] , self.y[idx])


