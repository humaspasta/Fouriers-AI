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

class Learning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(7 , 14, True)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(14, 7, True)
        self.loss_fn= nn.MSELoss()
    
    def forward(self , x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
    
    def training_step(self , batch , batchidx):
        x, y = batch
        y_pred = self(x)
        #need to find a proper way to extract actual values.
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss


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

points = []
for y in range(len(frame)):
    for x in range(len(frame[y])):
        if(frame[y, x][0] == 0):
            points.append((y,x))

points.sort(key=lambda x: (x[0] , x[1]) ,reverse=True)

top_half_points = [point for point in points if point[1] > 300]

top_half_x_points = [point[0] for point in top_half_points]
top_half_y_points = [point[1] for point in top_half_points]

bottom_half_points = [point for point in points if point[1] < 300]

bottom_half_x_points = [point[0] for point in bottom_half_points]
bottom_half_y_points = [point[1] for point in bottom_half_points]


x_in_order = top_half_x_points + bottom_half_x_points
y_in_order = bottom_half_y_points + bottom_half_y_points

coords_in_order = []
for i in range(len(x_in_order)):
    coords_in_order.append((x_in_order[i] , y_in_order[i]))


theta_in_order = np.arange(start=0 , stop=2 * math.pi , step=(2 * math.pi) / len(coords_in_order))

final_df = pd.DataFrame({
    'theta' : theta_in_order,
    'coords': coords_in_order,
    'X coords' : x_in_order,
    'Y coords' : y_in_order
})

#drop duplicate points
unique_df = final_df.groupby("X coords").first()

#creating a dataset for training
dataset = MyDataset(final_df['theta'] , final_df['coords'])

#training the AI



 









def reconstruct_tip(frequencies:tuple):
    '''
    Method for retrieving actual values.
    Works by tracing between 0 and 2pi and returning the resulting data.
    Resulting data will be used for calculating an error
    Error will be represented as the magnitude of a vector with each
    component of the vector containing an error for x and y. 
    Minimizing the mangitude of the error  vector is crucial to model success.
    
    '''
    circles = Drawing(600 , 600)
    actual_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    actual_frame = circles.draw_all_circles(one_rotation=True)#does a single rotation

    points = []
    for y in range(len(actual_frame)):
        for x in range(len(actual_frame[y])):
            if(frame[y, x][0] == 0):
                points.append((y,x))

    points.sort(key=lambda x: (x[0] , x[1]) ,reverse=True)

    top_half_points = [point for point in points if point[1] > 300]

    top_half_x_points = [point[0] for point in top_half_points]
    top_half_y_points = [point[1] for point in top_half_points]

    bottom_half_points = [point for point in points if point[1] < 300]

    bottom_half_x_points = [point[0] for point in bottom_half_points]
    bottom_half_y_points = [point[1] for point in bottom_half_points]


    x_in_order = top_half_x_points + bottom_half_x_points
    y_in_order = bottom_half_y_points + bottom_half_y_points

    coords_in_order = []
    for i in range(len(x_in_order)):
        coords_in_order.append((x_in_order[i] , y_in_order[i]))


    theta_in_order = np.arange(start=0 , stop=6 , step=6 / len(coords_in_order))

    actual_points = pd.DataFrame({
        'actual theta' : theta_in_order,
        'actual coords': coords_in_order,
        'actual X coords' : x_in_order,
        'actual y coords' : y_in_order
    })

    #drop duplicate points
    actual_points = final_df.groupby("X coords").first()

    return actual_points




    








