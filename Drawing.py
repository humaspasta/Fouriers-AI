import cv2
from CustomCircle import CustomCircle
import numpy as np
from Learning import Learning
import time

class Drawing:
    def __init__(self, length , width):
        self.length = length
        self.width = width
        self.frame = np.ones((self.length, self.width, 3), dtype=np.uint8) * 255


        self.circ = CustomCircle(None, 300 , 300, 100 , 0.02) # there is no frame initially. The frame is updated in the loop
        self.circ2 = CustomCircle(None , int(self.circ.calculate_rotate()[0]), int(self.circ.calculate_rotate()[1]), 30 , 0.05)
        self.circ3 = CustomCircle(None , int(self.circ2.calculate_rotate()[0]), int(self.circ2.calculate_rotate()[1]) , 10 , 0.07, isTip=True)


    def get_approximations():
        '''
        Function for retrieving learned results from machine learning algorithm
        '''
        pass

    def set_all_frames(self , frame):
        self.circ.set_frame(frame)
        self.circ2.set_frame(frame)
        self.circ3.set_frame(frame)

    def draw_all_circles(self):
        # Set up canvas
        width, height = 600, 600
        center = (width // 2, height // 2)
        radius = 150

        # Frame setup
        angle = 0
        fps = 60
        delay = 1 / fps
       
        frame_trace = np.ones((height, width, 3), dtype=np.uint8) * 255

        while True:
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            #frame = torch.ones(height , width, 3) * 255
            self.set_all_frames(frame)
            #tracing point

            # Create a white canvas
            self.circ.draw_circle()
            self.circ2.update_position(int(self.circ.calculate_rotate()[0]), int(self.circ.calculate_rotate()[1]))
            self.circ2.draw_circle()

            self.circ3.update_position(int(self.circ2.calculate_rotate()[0]), int(self.circ2.calculate_rotate()[1]))
            self.circ3.draw_circle()
            #cv2.circle(frame_trace, (int(self.circ3.calculate_rotate()[0]), int(self.circ3.calculate_rotate()[1])), radius=1, color=(0, 0, 0), thickness=-1)
            # Show frame

            cv2.imshow("Rotating Radius - Sin/Cos", frame)

            # Break with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(delay)

        cv2.destroyAllWindows()


    