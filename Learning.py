import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from MyDataset import MyDataset
import numpy as np
import cv2
from CustomCircle import CustomCircle
from Drawing_Module import Drawing
import pandas as pd
import matplotlib.pyplot as plt
import math
from pytorch_lightning import Trainer
import torch
from Sampling import DataProcessing


class Learning(pl.LightningModule):
    def __init__(self , num_circles=7):
        super().__init__()
        self.epoch_counter = 1
        self.freqs = torch.nn.Parameter(torch.rand(num_circles))
        self.layer1 = nn.Linear(7 , 14, True)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(14, 7, True)
        
    
    def forward(self):
        return self.freqs # returns final predicted frequencies
    
    def training_step(self , batch , batchidx):
        time_0 , x_0 , y_0 = batch #retrieving data from the batch
        predicted_omegas = self()
        sampler = DataProcessing()

        time , x , y = sampler.sample_frame(predicted_omegas)

        #Vectorize error:
        # Find indices where time matches time_0 exactly
        indices = (time == time_0).nonzero(as_tuple=True)[0]

        if indices.numel() == 0:
            # No exact match: find closest index
            diff = torch.abs(time - time_0)
            index = torch.argmin(diff)
        else:
            # Take first matching index if multiple found
            index = indices[0]

        loss_x = torch.nn.functional.mse_loss(x[index], x_0)
        loss_y = torch.nn.functional.mse_loss(y[index], y_0)
        loss = loss_x + loss_y

        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def on_train_epoch_end(self):
        # drawing = Drawing()
        # print("Starting drawing for epoch: " + " " + str(self.epoch_counter))
        # drawing.set_circle_omega(tuple(self.freqs))
        # drawing.draw_all_circles()
        return super().on_train_epoch_end()


##################################################################################################################################
'''
Actual Data processing here:

1. create a pattern and save it to an initially white numpy 3d matrix that represents the frame.
2. collect all points on that frame that have a color of zero (0,0,0)
3. sort the data with respect to X and if typbreakers occur do it with respect to y
4. add the points to a pandas Series
5. For points that are similar, take an average

'''

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



width, height = 600, 600

frame = np.ones((height, width, 3), dtype=np.uint8) * 255
cv2.circle(frame , (height//2, width//2),200, (0,0,0), thickness=1)

data_processor = DataProcessing() #sampling training data

times , XPoses , YPoses = data_processor.sample_circle()

#creating a dataset for training
dataset = MyDataset(times , XPoses, YPoses)

dataloader = DataLoader(dataset=dataset, batch_size=1)

trainer = Trainer(max_epochs=20)

model = Learning().to(device)

print("Training")
trainer.fit(model , dataloader)

output_freqs = model.freqs

draws = Drawing()

draws.set_circle_omega(tuple(output_freqs))

draws.draw_all_circles()

#training the Deep Learning algorithm



 













    








