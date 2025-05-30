import pygame
import math

# Initialize pygame
pygame.init()

# Set up display
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Rotating Radius with Sin/Cos Components")

# Colors
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE  = (0, 0, 255)
GRAY  = (200, 200, 200)
BLACK = (0,0,0)
# Circle parameters
center = (width // 2, height // 2)
radius = 150
angle = 0

clock = pygame.time.Clock()

running = True
while running:
    screen.fill(BLACK)
    
    # Handle quit event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw the unit circle
    pygame.draw.circle(screen, GRAY, center, radius, 2)

    # Calculate rotating point
    x = center[0] + radius * math.cos(angle)
    y = center[1] - radius * math.sin(angle)  # Minus because y increases downward in screen coords

    # Draw radius line
    pygame.draw.line(screen, RED, center, (x, y), 3)

    # Draw cosine (horizontal) projection
    pygame.draw.line(screen, BLUE, center, (x, center[1]), 2)

    # Draw sine (vertical) projection
    pygame.draw.line(screen, GREEN, (x, center[1]), (x, y), 2)

    pygame.draw.circle(screen , GRAY , (x,y), 15)

    # Update angle
    angle += 0.02
    if angle > 2 * math.pi:
        angle -= 2 * math.pi

    # Refresh screen
    pygame.display.flip()
    clock.tick(60)  # 60 FPS

pygame.quit()
