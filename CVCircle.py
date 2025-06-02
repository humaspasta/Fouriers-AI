import cv2
import numpy as np
import math
import time
from CustomCircle import CustomCircle


import torch

# Set up canvas
width, height = 600, 600
center = (width // 2, height // 2)
radius = 150

# Frame setup
angle = 0
fps = 60
delay = 1 / fps
circ = CustomCircle(None, 300 , 300, 100 , 0.02) # there is no frame initially. The frame is updated in the loop
circ2 = CustomCircle(None , int(circ.calculate_rotate()[0]), int(circ.calculate_rotate()[1]), 30 , 0.05)
circ3 = CustomCircle(None , int(circ2.calculate_rotate()[0]), int(circ2.calculate_rotate()[1]) , 10 , 0.07, isTip=True)
frame = np.ones((height, width, 3), dtype=np.uint8) * 255
frame_trace = np.ones((height, width, 3), dtype=np.uint8) * 255

while True:
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    #frame = torch.ones(height , width, 3) * 255
    circ.set_frame(frame)
    circ2.set_frame(frame)
    circ3.set_frame(frame)

    #tracing point

    # Create a white canvas
    circ.draw_circle()
    circ2.update_position(int(circ.calculate_rotate()[0]), int(circ.calculate_rotate()[1]))
    circ2.draw_circle()

    circ3.update_position(int(circ2.calculate_rotate()[0]), int(circ2.calculate_rotate()[1]))
    circ3.draw_circle()
    cv2.circle(frame_trace, (int(circ3.calculate_rotate()[0]), int(circ3.calculate_rotate()[1])), radius=1, color=(0, 0, 0), thickness=-1)
    # Show frame

    cv2.imshow("Rotating Radius - Sin/Cos", frame)

    # Break with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(delay)

cv2.destroyAllWindows()



    
    
    