import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from MyDataset import MyDataset
import numpy as np
import cv2
from CustomCircle import CustomCircle

class Learning(pl.LightningModule):
    def __init__(self , tip_circle:CustomCircle):
        super().__init__()
        self.layer1 = nn.Linear(7 , 14, True)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(14, 7, True)
        self.loss_fn= nn.MSELoss()
    
    def forward(self , x):
        self.layer1(x)
        self.activation(x)
        self.layer2(x)
        return x
    
    def training_step(self , batch , batchidx):
        x, y = batch
        y_pred = self(x)
        #need to find a proper way to extract actual values.
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss

width, height = 600, 600

frame = np.ones((height, width, 3), dtype=np.uint8) * 255
cv2.circle(frame , height//2, width//2, (200,0,0), 1)




data_loader = DataLoader(dataset, batch_size=16)

for epoch in range(5):
    for batch_x, batch_y in data_loader:
        preds = Learning(batch_x) # returns the predicted frequencies of rotation
        loss = Learning.loss_fn(preds, batch_y)
        loss.backward()





  
    

    

