import cv2
import numpy as np
from Drawing_Module import Drawing
import pandas as pd
import torch 

class DataProcessing:

    def __init__(self):
        self.device = None

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    # def sample_frame(self , omegas):
    #     '''
    #     Returns a dataframe of positions with a respective time value
    #     Frame: np dataframe used in open cv for drawing
    #     ''' 
    #     drawing = Drawing()

    #     list_of_times = drawing.draw_all_circles_once(omegas)
    #     times = []
    #     Xpos = []
    #     Ypos = []

    #     for time , x , y in list_of_times:
    #         times.append(time)
    #         Xpos.append(x)
    #         Ypos.append(y)
        
    #     times = torch.tensor(times, dtype=torch.float32).to(self.device)
    #     Xpos = torch.tensor(Xpos, dtype=torch.float32).to(self.device)
    #     Ypos = torch.tensor(Ypos, dtype=torch.float32).to(self.device)

    #     return times , Xpos , Ypos


    def sample_frame(self, omegas, radius=200, T=10, N=600):
        t = torch.linspace(0, T, N, device=omegas.device)
        x = torch.zeros(N, device=omegas.device)
        y = torch.zeros(N, device=omegas.device)

        for i, omega in enumerate(omegas):
            phase = 0  # you could also learn this
            r = radius / (i + 1)
            x += r * torch.cos(2 * torch.pi * omega * t + phase)
            y += r * torch.sin(2 * torch.pi * omega * t + phase)

        return t, x, y
    
    '''
    Sampling a circle for now
    '''
    def sample_circle(self, freq=1/10, radius=200, phase=0, T=10, N=10):
        t = torch.linspace(0, T, N)
        x = radius * torch.cos(2 * torch.pi * freq * t + phase)
        y = radius * torch.sin(2 * torch.pi * freq * t + phase)
       
       # Convert numpy arrays to PyTorch tensors (float32)
        t_tensor = torch.tensor(t, dtype=torch.float32).to(self.device)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        return t_tensor, x_tensor, y_tensor

