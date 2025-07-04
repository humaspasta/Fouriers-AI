import cv2
import math
class CustomCircle:
    def __init__(self, frame, x:int , y:int, radius:int, phase:int, angle_change:int, color=(0,0,0), isTip=False):
        self.x = x
        self.y = y
        self.frame = frame
        self.radius = radius
        self.angle = phase
        self.angle_change = angle_change
        self.isTip = isTip
        self.color = color
        

    def draw_circle(self):
        trace_points = []
        '''
        Draws circle along with rotating radius
        '''
        if self.frame is None:
            raise ValueError("Frame was not set")
        
        cv2.circle(self.frame, (self.x, self.y), self.radius, self.color, 1)

        x_rot = self.calculate_rotate()[0]
        y_rot = self.calculate_rotate()[1]

        # Draw radius line
        cv2.line(self.frame, (self.x, self.y), (int(x_rot) , int(y_rot)), 100, 1)
        


        self.angle += self.angle_change

        if self.angle > 2 * math.pi:
            self.angle -= 2 * math.pi

    def set_frame(self, frame):
        '''
        Sets the circles frame. Frame must be set before 
        '''
        self.frame = frame
        
    def get_center(self) -> tuple:
        '''
        Returns the center of the current circle
        '''
        return (self.x , self.y)
    
    def get_angle(self):
        return self.angle

    def calculate_rotate(self):
        '''
        Returns the position along the circle
        '''
        x_rot = self.x + self.radius * math.cos(self.angle)
        y_rot = self.y - self.radius * math.sin(self.angle)
        return (x_rot , y_rot)
    
    def set_omega(self, angle_change:float):
        self.angle_change = angle_change
    
    def set_radius(self , radius):
        self.radius = radius
    
    def set_color(self , color:tuple):
        self.color = color

    def trace(self):
        if self.isTip:
            self.frame[int(self.calculate_rotate()[1]) , int(self.calculate_rotate()[0])] = (0,0,0)
            self.set_frame(self.frame)

    def get_frame(self):
        return self.frame
    
    def update_position(self , x , y):
        '''
        Updates the position of the circle
        '''
        self.x = x
        self.y = y
    
    def set_phase(self , phase):
        self.angle = phase

    

            
        
