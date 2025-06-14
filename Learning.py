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
import torch.nn.functional as F


class Learning(pl.LightningModule):
    def __init__(self , num_circles=7):
        super().__init__()
        self.epoch_counter = 1
        self.freqs = torch.nn.Parameter(torch.rand(num_circles))
        self.radii = torch.nn.Parameter(torch.rand(num_circles) * 100)
        self.phases = torch.nn.Parameter(torch.rand(num_circles) * 2 * torch.pi)
        self.layer1 = nn.Linear(7 , 14, True)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(14, 7, True)
        
    
    def forward(self):
        return self.freqs, self.radii, self.phases # returns final predicted frequencies
    
    def training_step(self , batch , batchidx):
        time_0 , x_0 , y_0 = batch #retrieving data from the batch
        predicted_omegas, predicted_radii, predicted_phases = self()
        sampler = DataProcessing()

        time , x , y = sampler.sample_frame(predicted_omegas, predicted_radii , predicted_phases, N=len(time_0))

        # indices = (time == time_0).nonzero(as_tuple=True)[0]

        # if indices.numel() == 0:
        #     # No exact match: find closest index
        #     diff = torch.abs(time - time_0)
        #     index = torch.argmin(diff)
        # else:
        #     # Take first matching index if multiple found
        #     index = indices[0]

        #error is now in terms of euclidian distance
        loss = F.mse_loss(torch.sqrt(x**2 + y**2) , torch.sqrt(x_0**2 + y_0**2))
        self.log("train_loss", loss)
        print("total loss: " + str(loss))
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

dataloader = DataLoader(dataset=dataset, batch_size=20)

trainer = Trainer(max_epochs=10000)

model = Learning().to(device)

print("Training")
trainer.fit(model , dataloader)

output_freqs = model.freqs
output_phase = model.phases
output_radii = model.radii.int()

print("Output radii here: " + str(output_radii))


draws = Drawing()

print(output_freqs , output_phase, output_radii)

draws.set_circle_omega(tuple(output_freqs.detach().cpu().numpy()/10))
draws.set_all_radius(tuple(output_radii.detach().cpu().numpy()))
draws.set_all_phase(tuple(output_phase.detach().cpu().numpy()))



draws.draw_all_circles()

#training the Deep Learning algorithm



 













    








