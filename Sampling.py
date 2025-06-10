import cv2
import numpy as np


class DataProcessing:
    def  __init__(self, frame_to_sample , time_to_sample=10):
        self.time_to_sample = time_to_sample
        self.frame = frame_to_sample

    def sample_frame(self):
        pass
