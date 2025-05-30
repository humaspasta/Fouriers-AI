import pygame
import math
class CustomCircle:

    def __init__(self, screen:pygame.Surface, x:int , y:int, radius:int):
        self.x = x
        self.y = y
        self.screen = screen

    
    def draw_circle(self):
        pygame.draw.circle(self.screen, (255,255,255), (self.x,self.y))
        
    
