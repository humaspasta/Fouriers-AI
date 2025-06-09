import torch
from torch.utils.data import Dataset
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, theta:pd.Series, x_points:pd.Series , y_points:pd.Series):
        '''
        Data: The actual output from the model
        Lables: The expected output
        '''
        if len(theta) != len(x_points) or len(theta) != len(y_points):
            raise ValueError("dataframe sizes are not the same")
        

        self.theta = torch.tensor(theta) #scalars representing overall rotation angle
        self.x = torch.tensor(x_points) # tensor representing position x given theta
        self.y = torch.tensor(y_points) #tensor representing position y given theta

    def __len__(self): 
        return len(self.theta)

    def __getitem__(self, idx):
        return (self.theta.iloc[idx] , self.x.iloc[idx] , self.y.iloc[idx] , self.theta)


