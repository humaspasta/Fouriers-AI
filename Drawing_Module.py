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
        self.circ2 = CustomCircle(None , int(self.circ.calculate_rotate()[0]), int(self.circ.calculate_rotate()[1]), 100 , 0.05)
        self.circ3 = CustomCircle(None , int(self.circ2.calculate_rotate()[0]), int(self.circ2.calculate_rotate()[1]) , 50 , 0.07)
        self.circ4 = CustomCircle(None , int(self.circ3.calculate_rotate()[0]), int(self.circ3.calculate_rotate()[1]), 25, 0.03)
        self.circ5 = CustomCircle(None , int(self.circ4.calculate_rotate()[0]), int(self.circ4.calculate_rotate()[1]) , 12 , 0.05)
        self.circ6 = CustomCircle(None , int(self.circ5.calculate_rotate()[0]), int(self.circ5.calculate_rotate()[1]), 6, 0.01)
        self.circ7 = CustomCircle(None , int(self.circ6.calculate_rotate()[0]), int(self.circ6.calculate_rotate()[1]), 3 , 0.02 , isTip=True)


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
    
    def set_circle_frequencies(self , frequencies:tuple):
        freq1, freq2, freq3, freq4, freq5, freq6, freq7 = frequencies
        self.circ.set_frequency(freq1)
        self.circ2.set_frequency(freq2)
        self.circ3.set_frequency(freq3)
        self.circ4.set_frequency(freq4)
        self.circ5.set_frequency(freq5)
        self.circ6.set_frequency(freq6)
        self.circ7.set_frequency(freq7)

    def draw_all_circles_once(self , theta_points , frequencies) -> tuple:
        '''
        Draws with a single rotation
        '''
        width, height = 600, 600
        # Frame setup
        angle = 0
        fps = 60
        delay = 1 / fps
       
        frame_trace = np.ones((height, width, 3), dtype=np.uint8) * 255
        x_0 , y_0 = self.circ.get_center()
        if theta_points == None:
            raise ValueError("theta_points is null")
        
        self.set_circle_frequencies(tuple(frequencies))
        similar_points = []

        while round(self.calculate_current_theta(x_0 , y_0) , 3) < 2 * math.pi:
            
            if round(self.calculate_current_theta(x_0 , y_0) , 3) in theta_points:
                similar_points.append((round(self.calculate_current_theta(x_0 , y_0) , 3),
                                       int(self.circ5.calculate_rotate()[0]) , 
                                       int(self.circ5.calculate_rotate()[1])))
            

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

        return similar_points #returns a tuple of lists. the first list contains theta points , the second contains x and y points

    

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

            self.circ4.update_position(int(self.circ3.calculate_rotate()[0]), int(self.circ3.calculate_rotate()[1]))
            self.circ4.draw_circle()

            self.circ5.update_position(int(self.circ4.calculate_rotate()[0]), int(self.circ4.calculate_rotate()[1]))
            self.circ5.draw_circle()

            self.circ6.update_position(int(self.circ5.calculate_rotate()[0]), int(self.circ5.calculate_rotate()[1]))
            self.circ6.draw_circle()

            self.circ7.update_position(int(self.circ6.calculate_rotate()[0]), int(self.circ6.calculate_rotate()[1]))
            self.circ7.draw_circle()

            cv2.imshow("Rotating Radius - actual", frame)

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



    