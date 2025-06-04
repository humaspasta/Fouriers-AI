import cv2
from CustomCircle import CustomCircle
import numpy as np
import time
import math

class Drawing:
    def __init__(self, length , width):
        self.length = length
        self.width = width
        self.frame = np.ones((self.length, self.width, 3), dtype=np.uint8) * 255

        self.circ = CustomCircle(None, 300 , 300, 100 , 0.02) # there is no frame initially. The frame is updated in the loop
        self.circ2 = CustomCircle(None , int(self.circ.calculate_rotate()[0]), int(self.circ.calculate_rotate()[1]), 50 , 0.05)
        self.circ3 = CustomCircle(None , int(self.circ2.calculate_rotate()[0]), int(self.circ2.calculate_rotate()[1]) , 25 , 0.07)
        self.circ4 = CustomCircle(None , int(self.circ3.calculate_rotate()[0]), int(self.circ3.calculate_rotate()[1]), 12, 0.03)
        self.circ5 = CustomCircle(None , int(self.circ3.calculate_rotate()[0]), int(self.circ3.calculate_rotate()[1]) , 6 , 0.05, isTip = True)


    def get_approximations():
        '''
        Function for retrieving learned results from machine learning algorithm
        '''
        pass

    def set_all_frames(self , frame):
        self.circ.set_frame(frame)
        self.circ2.set_frame(frame)
        self.circ3.set_frame(frame)
        self.circ4.set_frame(frame)
        self.circ5.set_frame(frame)
    
    def set_tip_frame(self , frame):
        self.circ5.set_frame(frame)

    def get_tip_frame(self):
        return self.circ5.get_frame()
    
    def set_circle_frequencies(self , frequencies:tuple):
        freq1, freq2, freq3, freq4, freq5 = frequencies

        self.circ.set_frequency(freq1)
        self.circ2.set_frequency(freq2)
        self.circ3.set_frequency(freq3)
        self.circ4.set_frequency(freq4)
        self.circ5.set_frequency(freq5)

    

    def draw_all_circles(self, one_rotation=False):
        width, height = 600, 600
        center = (width // 2, height // 2)
        radius = 150

        # Frame setup
        angle = 0
        fps = 60
        delay = 1 / fps
       
        frame_trace = np.ones((height, width, 3), dtype=np.uint8) * 255

        if(one_rotation):
            x_0 , y_0 = self.circ.get_center()
            
            while self.calculate_current_theta(x_0 , y_0) < 2 * math.pi:
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

                self.circ4.update_position(int(self.circ3.calculate_rotate()[0]), int(self.circ3.calculate_rotate()[1]))
                self.circ4.draw_circle()

                self.circ5.update_position(int(self.circ4.calculate_rotate()[0]), int(self.circ4.calculate_rotate()[1]))
                self.circ5.draw_circle()

                #point circle
                cv2.circle(frame_trace , (), 1, (0,0,0), 1, cv2.LINE_AA) # point that traces values of overall rotation on second frame

                cv2.imshow("Rotating Radius - Training", frame)

                # Break with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(delay)
        else:
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

                self.circ4.update_position(int(self.circ3.calculate_rotate()[0]), int(self.circ3.calculate_rotate()[1]))
                self.circ4.draw_circle()

                self.circ5.update_position(int(self.circ4.calculate_rotate()[0]), int(self.circ4.calculate_rotate()[1]))
                self.circ5.draw_circle()

                cv2.imshow("Rotating Radius - actual", frame)

                # Break with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(delay)

        cv2.destroyAllWindows()
        return frame_trace

    
    def calculate_current_theta(self, x_0 , y_0):
        x , y = self.circ5.calculate_rotate()
        return math.asin(y - y_0 / ((y-y_0)**2 + (x-x_0)**2))



    