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
    def __init__(self , num_circles):
        super().__init__()
        self.paramters = torch.nn.Parameter(torch.rand(num_circles))
        self.layer1 = nn.Linear(7 , 14, True)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(14, 7, True)
        self.loss_fn= lambda x,y : math.sqrt(x**2 + y**2)
    
    def forward(self , x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x # returns final predicted frequencies
    
    
    def reconstruct_tip(frequencies:tuple , theta_points):
        '''
        Method for retrieving actual values.
        Works by tracing between 0 and 2pi and returning the resulting data.
        Will only sample points that are already in theta_points iterable argument.
        '''
        circles = Drawing(600 , 600)
        
        points = circles.draw_all_circles_once(theta_points , frequencies)#does a single rotation

        points.sort(key=lambda x: (x[0]) )

        top_half_points = [point for point in points if point[1] > 300]

        top_half_x_points = [point[0] for point in top_half_points]
        top_half_y_points = [point[1] for point in top_half_points]

        bottom_half_points = [point for point in points if point[1] < 300]

        bottom_half_x_points = [point[0] for point in bottom_half_points]
        bottom_half_y_points = [point[1] for point in bottom_half_points]


        x_in_order = top_half_x_points + bottom_half_x_points
        y_in_order = top_half_y_points + bottom_half_y_points

        coords_in_order = []
        for i in range(len(x_in_order)):
            coords_in_order.append((x_in_order[i] , y_in_order[i]))


        theta_in_order = np.arange(start=0 , stop=6 , step=6 / len(coords_in_order))

        actual_points = pd.DataFrame({
            'actual theta' : theta_in_order,
            'actual coords': coords_in_order,
            'actual X coords' : x_in_order,
            'actual Y coords' : y_in_order
        })

        #drop duplicate points
        actual_points = final_df.groupby("X coords").first()

        return actual_points
    

    #revise so that we are able to backpropogate. We must find another way of retrieving expected x and y values given a theta. 
    def training_step(self , batch , batchidx):
        theta , x_0 , y_0, theta_points = batch #retrieving data from the batch
        predicted_frequencies = self(x)

        predicted_dataframe = self.reconstruct_tip(predicted_frequencies , theta_points)
        predicted_dataframe.groupby('actual_theta')
        x , y = predicted_dataframe.loc[theta]

        #Vectorize error:
        error_vector = [x - x_0 , y - y_0]

        loss = self.loss_fn(error_vector[0]**2  , error_vector[1]**2)
        self.log("train_loss", loss)
        return loss



##################################################################################################################################
'''
Actual Data processing here:

1. create a pattern and save it to an initially white numpy 3d matrix that represents the frame.
2. collect all points on that frame that have a color of zero (0,0,0)
3. sort the data with respect to X and if typbreakers occur do it with respect to y
4. add the points to a pandas Series
5. For points that are similar, take an average

'''
width, height = 600, 600

frame = np.ones((height, width, 3), dtype=np.uint8) * 255
cv2.circle(frame , (height//2, width//2),200, (0,0,0), thickness=1)

data_processor = DataProcessing(frame , 10)

labels = data_processor.sample_frame()


#creating a dataset for training
dataset = MyDataset(labels['theta'] , labels['X Coords'], labels['Y Coords'])

dataloader = DataLoader(dataset=dataset, batch_size=1)

trainer = Trainer(max_epochs=20)

trainer.fit(Learning , dataloader)
#training the Deep Learning algorithm



 













    








