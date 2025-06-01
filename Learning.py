import pytorch_lightning as pl
import torch
from torch import nn


class Learning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(7 , 14, True)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(14, 7, True)
        self.acitvation2 = nn.ReLU()
    
    def forward(self , x):
        self.layer1(x)
        self.activation(x)
        self.layer2(x)
        return x
    
    def training_step(self , batch , batchidx):
        x, y = batch 
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss
    def on_train_epoch_end(self):
        #drawing code here
        print("Epoch complete")
    
        

