import os
import sys
import cv2
import pygame
import numpy as np
from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3

pygame.init()
pygame.font.init()

# Get the display dimensions
screen_info = pygame.display.Info()
screen_width = 0.8*screen_info.current_w
screen_height = 0.8*screen_info.current_h

# Set up the screen
screen = pygame.display.set_mode((screen_width, screen_height))

# Window and font setup
pygame.display.set_caption("Gaze tracking test")
font_size = 48
bold_font = pygame.font.Font(None, font_size)
bold_font.set_bold(True)

# Set up colors
RED = (255, 0, 100)
BLUE = (100, 0, 255)
GREEN = (0, 255, 0)
BLANK = (0,0,0)
WHITE = (255, 255, 255)

# Initialize video and gestures
gestures = EyeGestures_v3()
video_capture = VideoCapture(0)

# Create calibration grid
x = np.arange(0, 1.1, 0.2)
y = np.arange(0, 1.1, 0.2)

xx, yy = np.meshgrid(x, y)

# Set up calibration map
calibration_map = np.column_stack([xx.ravel(), yy.ravel()])
n_points = min(len(calibration_map),25)
np.random.shuffle(calibration_map)
gestures.uploadCalibrationMap(calibration_map,context="my_context")
gestures.setFixation(1.0)

# Initialize clock
clock = pygame.time.Clock()

# Main game loop
running = True
iterator = 0
prev_x = 0
prev_y = 0
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False         
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL:
                running = False
            if event.key == pygame.K_ESCAPE:
                running = False

    try:
        ret, frame = video_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.flip(frame, axis=1)

        calibrate = (iterator <= n_points) # Calibrate for the first n_points
        event, calibration = gestures.step(frame, calibrate, screen_width, screen_height, context="my_context")

        if event is None:
            continue

        # Prepare the screen for a new frame
        screen.fill(BLANK)

        if event is not None or calibration is not None:
            # Display frame on Pygame screen
            screen.blit(
                pygame.surfarray.make_surface(np.rot90(event.sub_frame)),
                (0, 0)
            )
            my_font = pygame.font.SysFont('Comic Sans MS', 30)
            text_surface = my_font.render(f'{event.fixation}', False, (0, 0, 0))
            screen.blit(text_surface, (0,0))

            if calibrate:
                if calibration.point[0] != prev_x or calibration.point[1] != prev_y:
                    iterator += 1
                    prev_x = calibration.point[0]
                    prev_y = calibration.point[1]
                # pygame.draw.circle(screen, GREEN, fit_point, calibration_radius)
                pygame.draw.circle(screen, BLUE, calibration.point, calibration.acceptance_radius)
                text_surface = bold_font.render(f"{iterator}/{n_points}", True, WHITE)
                text_square = text_surface.get_rect(center=calibration.point)
                screen.blit(text_surface, text_square)
            
            if gestures.whichAlgorithm(context="my_context") == "Ridge":
                pygame.draw.circle(screen, RED, event.point, 50)
            if gestures.whichAlgorithm(context="my_context") == "LassoCV":
                pygame.draw.circle(screen, BLUE, event.point, 50)
            if event.saccades:
                pygame.draw.circle(screen, GREEN, event.point, 50)

            my_font = pygame.font.SysFont('Comic Sans MS', 30)
            text_surface = my_font.render(f'{gestures.whichAlgorithm(context="my_context")}', False, (0, 0, 0))
            screen.blit(text_surface, event.point)
            
    except Exception as e:
        # Catch any error from gestures.step() or drawing logic
        print(f"An error occurred: {e}")
        # Optionally display an error on screen
        error_text = bold_font.render("An error occurred. Check console.", True, RED)
        text_rect = error_text.get_rect(center=(screen_width // 2, screen_height // 2))
        screen.blit(error_text, text_rect)
        pass # Continue to the next frame

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()