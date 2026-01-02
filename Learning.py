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
        self.freqs = torch.nn.Parameter(torch.rand(num_circles))
        self.radii = torch.nn.Parameter(torch.rand(num_circles) * 100)
        self.phases = torch.nn.Parameter(torch.rand(num_circles) * 2 * torch.pi)
        self.ki = torch.tensor(0.0)
        self.kp = torch.tensor(0.0)
        self.kd = torch.tensor(0.0)
      
        
    
    def forward(self):
        return self.freqs, self.radii, self.phases # returns final predicted frequencies
    
    def training_step(self , batch , batchidx):
        time_0 , x_0 , y_0 = batch #retrieving data from the batch
        predicted_omegas, predicted_radii, predicted_phases = self()
        sampler = DataProcessing()

        time , x , y = sampler.sample_frame(predicted_omegas, predicted_radii , predicted_phases, N=len(time_0))

        #error is now in terms of manhattan distance
        
        loss = self.get_PID_Error(x, x_0 , y , y_0)

        self.log("train_loss", loss)
        print("total loss: " + str(loss))
        return loss
    
    def get_PID_Error(self, x:torch.tensor, x_0:torch.tensor, y:torch.tensor, y_0:torch.tensor):


        e = torch.sqrt(torch.pow(x - x_0 , 2) + torch.pow(y - y_0 , 2))
        self.kp = e.mean()

        # self.ki  = self.ki.detach() +self.kp 
        
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        delta_s = torch.sqrt(dx**2 + dy**2)

        de = e[1:] - e[:-1]             # shape [N-1]
        D = de / (delta_s + 1e-8)       # add small eps to prevent div by 0

        self.kd = D.mean()

        return self.kp + self.kd


    
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

times , XPoses , YPoses = data_processor.sample_circle(N=100)


#creating a dataset for training
dataset = MyDataset(times , XPoses, YPoses)

dataloader = DataLoader(dataset=dataset, batch_size=100)

trainer = Trainer(max_epochs=50000)

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


# #training the Deep Learning algorithm



 













#     # import tensorflow as tf
# # from CustomCircle import CustomCircle
# # from Sampling import DataProcessing
# # from MyDataset import MyDataset
# # from Drawing_Module import Drawing
# # import keras
# # from keras import layers




# # class deep_learning():
# #     def __init__(self):
# #         self.model = keras.Sequential()
    
# #     def create_model(self):
# #         self.model.add(layers.Input(shape=(7,)),
# #                        layers.Dense(128, activation='ReLU'),
# #                        layers.Dense(64, activation='ReLU'),
# #                        layers.Dropout(0.2),
# #                        layers.Dense(32, activation='ReLU'),
# #                        layers.Dense(14,))

# #     def train_model(self):
# #         self.model.fit()



# '''
# Output radii here: tensor([25, 82, 34, 58,  9, 66, 61], dtype=torch.int32)
# Parameter containing:
# tensor([ 0.1005, -0.0997,  0.0998,  0.1003,  0.2815,  0.2815,  0.2815],
#        requires_grad=True) Parameter containing:
# tensor([6.2376, 7.8043, 6.1651, 0.0193, 5.8348, 3.6157, 6.8794],
#        requires_grad=True) tensor([25, 82, 34, 58,  9, 66, 61], dtype=torch.int32)
# #results in the drawing of an ellipse of exact radius of circle
# '''

'''
Output radii here: tensor([55, 17, 63, 81, 38, 24, 30], dtype=torch.int32)
Parameter containing:
tensor([0.1006, 0.5998, 0.1004, 0.0994, 0.5608, 0.5927, 0.5564],
       requires_grad=True) Parameter containing:
tensor([-0.0100,  9.7299,  6.2770,  0.0108, -1.5660,  0.5405,  1.7178],
       requires_grad=True) tensor([55, 17, 63, 81, 38, 24, 30], dtype=torch.int32)
#output of model with pid error
'''