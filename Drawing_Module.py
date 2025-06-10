import cv2
from CustomCircle import CustomCircle
import numpy as np
import time
import math
import random

class Drawing:
    def __init__(self, length=600 , width=600):
        self.length = length
        self.width = width
        self.frame = np.ones((self.length, self.width, 3), dtype=np.uint8) * 255

        self.circ = CustomCircle(None, 300 , 300, 100 , 0.02) # there is no frame initially. The frame is updated in the loop
        self.circ2 = CustomCircle(None , int(self.circ.calculate_rotate()[0]), int(self.circ.calculate_rotate()[1]), 100 , 0.01)
        self.circ3 = CustomCircle(None , int(self.circ2.calculate_rotate()[0]), int(self.circ2.calculate_rotate()[1]) , 50 , 0.05)
        self.circ4 = CustomCircle(None , int(self.circ3.calculate_rotate()[0]), int(self.circ3.calculate_rotate()[1]), 25, 0.07)
        self.circ5 = CustomCircle(None , int(self.circ4.calculate_rotate()[0]), int(self.circ4.calculate_rotate()[1]) , 12 , random.random())
        self.circ6 = CustomCircle(None , int(self.circ5.calculate_rotate()[0]), int(self.circ5.calculate_rotate()[1]), 6, 0.1)
        self.circ7 = CustomCircle(None , int(self.circ6.calculate_rotate()[0]), int(self.circ6.calculate_rotate()[1]), 3 , 0.03 , isTip=True)


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
        self.circ6.set_frame(frame)
        self.circ7.set_frame(frame)
    
    def set_tip_frame(self , frame):
        self.circ5.set_frame(frame)

    def get_tip_frame(self):
        return self.circ5.get_frame()
    
    def set_circle_omega(self , frequencies:tuple):
        omega1, omega2, omega3, omega4, omega5, omega6, omega7 = frequencies
        self.circ.set_omega(omega1)
        self.circ2.set_omega(omega2)
        self.circ3.set_omega(omega3)
        self.circ4.set_omega(omega4)
        self.circ5.set_omega(omega5)
        self.circ6.set_omega(omega6)
        self.circ7.set_omega(omega7)

    def draw_all_circles_once(self , omegas) -> tuple:
        '''
        Draws with a single rotation. Draws with respect to time and returns the frame
        '''
        width, height = 600, 600
        # Frame setup
        angle = 0
        fps = 60
        delay = 1 / fps
        curr_time = 0
       
        frame_trace = np.ones((height, width, 3), dtype=np.uint8) * 255
        x_0 , y_0 = self.circ.get_center()
        
        self.set_circle_omega(tuple(omegas))
        timed_points = []
        while curr_time <= 10:
            
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

            self.circ6.update_position(int(self.circ5.calculate_rotate()[0]), int(self.circ5.calculate_rotate()[1]))
            self.circ6.draw_circle()

            self.circ7.update_position(int(self.circ6.calculate_rotate()[0]), int(self.circ6.calculate_rotate()[1]))

            self.circ7.draw_circle()
            #point circle to trace on result frame
            cv2.circle(frame_trace , (int(self.circ7.calculate_rotate()[0]) , int(self.circ7.calculate_rotate()[1])), 1, (0,0,0), 1, cv2.LINE_AA) # point that traces values of overall rotation on second frame

            cv2.circle(frame, (height//2, width//2),200, (255,255,0), thickness=1)


            time.sleep(delay)

            timed_points.append((curr_time , int(self.circ7.calculate_rotate()[0]) , int(self.circ7.calculate_rotate()[1])))
            curr_time += delay

        return timed_points #returns a tuple of lists. the first list contains theta points , the second contains x and y points

    

    def draw_all_circles(self):
        '''
        Draws all circles with frequencies
        '''
        width, height = 600, 600
        center = (width // 2, height // 2)
        radius = 150

        # Frame setup
        angle = 0
        fps = 60
        delay = 1 / fps
        total_frame = np.ones((height, width, 3), dtype=np.uint8) * 255

        frame_trace = np.ones((height, width, 3), dtype=np.uint8) * 255
        point_circle = CustomCircle(frame_trace , 0, 0, 1, 0, (0,0,255))
        
        background = np.ones((height, width, 3), dtype=np.uint8) * 255

        
        while True:
            #tracing point
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            #frame = torch.ones(height , width, 3) * 255
            self.set_all_frames(frame)
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

            self.circ6.update_position(int(self.circ5.calculate_rotate()[0]), int(self.circ5.calculate_rotate()[1]))
            self.circ6.draw_circle()

            self.circ7.update_position(int(self.circ6.calculate_rotate()[0]), int(self.circ6.calculate_rotate()[1]))
            self.circ7.draw_circle()

            point_circle.update_position(int(self.circ7.calculate_rotate()[0]), int(self.circ7.calculate_rotate()[1]))
            point_circle.draw_circle()

            cv2.circle(frame , (300 , 300) , 200 , (255,255,0), 1)


            combined = cv2.addWeighted(frame_trace, 0.5, frame, 0.5, 0)
        
            cv2.imshow("Rotating Radius - actual", combined)


            # Break with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(delay)

        cv2.destroyAllWindows()
      

    
    def calculate_current_theta(self, x_0 , y_0):
        '''
        Returns a global value of theta in the rotation
        '''
        x , y = self.circ5.calculate_rotate()

        dx = x - x_0
        dy = y - y_0

        return math.atan2(dx,dy)



    