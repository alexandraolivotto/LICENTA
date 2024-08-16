import numpy as np
import cv2 as cv

# Function to calculate the angle between three joints
class Utils:
    fps = 30

    ORANGE_SHADE = (235, 94, 40)
    GRAY_SHADE = (64, 61, 57)
    WHITE_SHADE = (255, 252, 242)

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
            (body.left_elbow_angle, body.left_elbow),
            (body.right_elbow_angle, body.right_elbow),
            (body.left_shoulder_angle, body.left_shoulder),
            (body.right_shoulder_angle, body.right_shoulder),
            # (body.left_knee_angle, body.left_knee),
            # (body.right_knee_angle, body.right_knee),
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
