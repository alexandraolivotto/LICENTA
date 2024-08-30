import numpy as np
import cv2 as cv
import pygame

from sprite import AnimatedSprite


# Function to calculate the angle between three joints
class Utils:
    width = 1280
    height = 1024
    fps = 30
    font = './resources/fonts/Laro Soft Medium.ttf'
    BLUE_SHADE_HOVER = (33, 164, 176)
    BLUE_SHADE_DARK = (25, 135, 145)
    BLUE_SHADE_BRIGHT = (30, 201, 217)
    GRAY_SHADE = (64, 61, 57)
    GRAY_SHADE_HOVER = (109, 105, 99)
    WHITE_SHADE = (255, 252, 242)
    BLACK = (0, 0, 0)

    EDGES = {
        (0, 1): 'm',
        (0, 2): 'c',
        (1, 3): 'm',
        (2, 4): 'c',
        (0, 5): 'm',
        (0, 6): 'c',
        (5, 7): 'm',
        (7, 9): 'm',
        (6, 8): 'c',
        (8, 10): 'c',
        (5, 6): 'y',
        (5, 11): 'm',
        (6, 12): 'c',
        (11, 12): 'y',
        (11, 13): 'm',
        (13, 15): 'm',
        (12, 14): 'c',
        (14, 16): 'c'
    }

    landmark_dict = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16,
    }

    @staticmethod
    def display_angles(frame, body):
        angles = [
            # (body.left_elbow_angle, body.left_elbow),
            # (body.right_elbow_angle, body.right_elbow),
            # (body.left_shoulder_angle, body.left_shoulder),
            # (body.right_shoulder_angle, body.right_shoulder),
            # (body.left_knee_angle, body.left_knee),
            (body.right_knee_angle, body.right_knee),
            # # (body.left_outer_hip_angle, body.left_hip),
            # (body.left_inner_hip_angle, body.left_hip),
            # # (body.right_outer_hip_angle, body.right_hip),
            # (body.right_inner_hip_angle, body.right_hip)
        ]

        for angle, part in angles:
            Utils.put_text_on_frame(frame, angle, part)

    @staticmethod
    def put_text_on_frame(frame, body_angle, body_part):
        h, w, c = frame.shape
        cv.putText(frame, str(round(body_angle, 2)),
                   tuple(np.multiply(np.flip(body_part[:2]), [w, h]).astype(int)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA
                   )

    @staticmethod
    def get_gif_from_url(image_url, x=200, y=height - 50, scale=1):
        animation_frame_list = AnimatedSprite.loadGIF(image_url, scale)
        animated_sprite = AnimatedSprite(x, y, animation_frame_list)
        gif = pygame.sprite.Group(animated_sprite)
        return gif
