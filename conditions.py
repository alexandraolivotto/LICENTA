import numpy as np


def verify_rep(percentage, direction):
    if percentage == 100:
        if direction == 0:
            print("Half there")
            return False, 1
    if percentage == 0:
        if direction == 1:
            print("Rep completed")
            return True, 0

    return False, direction


def verify_rep_double_sided_exercises(percentage, direction):
    if percentage == 100:
        if direction == 0:
            print("1/4 there")
            return False, 1
        if direction == 2:
            print("3/4 there")
            return False, 3
    if percentage == 0:
        if direction == 1:
            print("half there")
            return False, 2
        if direction == 3:
            print("Rep completed")
            return True, 0
    return False, direction


def left_elbow_bend(body, direction):
    percentage = np.interp(body.right_elbow_angle, (90, 175), (0, 100))
    return verify_rep(percentage, direction)


def crunches(body, direction):
    percentage = np.interp(body.right_outer_hip_angle, (55, 145), (0, 100))
    return verify_rep(percentage, direction)


def pile_squats(body, direction):
    left_knee_percentage = np.interp(body.left_knee_angle, (90, 175), (0, 100)) * 0.5
    right_knee_percentage = np.interp(body.right_knee_angle, (90, 175), (0, 100)) * 0.5
    percentage = int(left_knee_percentage + right_knee_percentage)
    return verify_rep(percentage, direction)


def lunges_right_leg(body, direction):
    left_knee_percentage = np.interp(body.left_knee_angle, (90, 175), (100, 0)) * 0.5
    right_knee_percentage = np.interp(body.right_knee_angle, (110, 175), (100, 0)) * 0.5
    percentage = int(left_knee_percentage + right_knee_percentage)
    return verify_rep(percentage, direction)


def lunges_left_leg(body, direction):
    left_knee_percentage = np.interp(body.left_knee_angle, (110, 175), (100, 0)) * 0.5
    right_knee_percentage = np.interp(body.right_knee_angle, (90, 175), (100, 0)) * 0.5
    percentage = int(left_knee_percentage + right_knee_percentage)
    return verify_rep(percentage, direction)


def side_lunges_right_leg(body, direction):
    left_knee_percentage = np.interp(body.left_knee_angle, (100, 175), (100, 0)) * 0.8
    right_knee_percentage = np.interp(body.right_knee_angle, (174, 177), (100, 0)) * 0.2
    percentage = int(left_knee_percentage + right_knee_percentage)
    return verify_rep(percentage, direction)


def side_lunges_left_leg(body, direction):
    right_knee_percentage = np.interp(body.right_knee_angle, (100, 175), (100, 0)) * 0.8
    left_knee_percentage = np.interp(body.left_knee_angle, (174, 177), (100, 0)) * 0.2
    percentage = int(right_knee_percentage + left_knee_percentage)
    return verify_rep(percentage, direction)


def squats(body, direction):
    right_knee_percentage = np.interp(body.right_knee_angle, (65, 175), (100, 0)) * 0.5
    right_shoulder_percentage = np.interp(body.right_shoulder_angle, (90, 110), (0, 100)) * 0.5
    percentage = int(right_knee_percentage + right_shoulder_percentage)
    return verify_rep(percentage, direction)


def hip_stretch_left_leg(body, direction):
    right_knee_percentage = np.interp(body.right_knee_angle, (45, 90), (100, 0)) * 0.5
    left_knee_percentage = np.interp(body.left_knee_angle, (90, 130), (0, 100)) * 0.5
    percentage = int(left_knee_percentage + right_knee_percentage)
    return verify_rep(percentage, direction)


def hip_stretch_right_leg(body, direction):
    left_knee_percentage = np.interp(body.left_knee_angle, (45, 90), (100, 0)) * 0.5
    right_knee_percentage = np.interp(body.right_knee_angle, (90, 130), (0, 100)) * 0.5
    percentage = int(left_knee_percentage + right_knee_percentage)
    return verify_rep(percentage, direction)


def good_morning_stretch(body, direction):
    right_outer_hip_percentage = np.interp(body.right_outer_hip_angle, (90, 175), (100, 0)) * 0.5
    right_knee_percentage = np.interp(body.right_knee_angle, (150, 175), (100, 0)) * 0.5
    percentage = int(right_outer_hip_percentage + right_knee_percentage)
    print(percentage)
    return verify_rep(percentage, direction)


def press_up_back(body, direction):
    right_outer_hip_percentage = np.interp(body.right_outer_hip_angle, (160, 175), (100, 0))
    return verify_rep(int(right_outer_hip_percentage), direction)


def left_leg_elevation(body, direction):
    right_outer_hip_percentage = np.interp(body.right_outer_hip_angle, (160, 175), (100, 0))
    return verify_rep(int(right_outer_hip_percentage), direction)


def right_leg_elevation(body, direction):
    left_outer_hip_percentage = np.interp(body.left_outer_hip_angle, (160, 175), (100, 0))
    return verify_rep(int(left_outer_hip_percentage), direction)


#double sided exercises

def side_bends(body, direction):
    #user should always start with left side
    right_shoulder_percentage = np.interp(body.right_shoulder_angle, (160, 175), (100, 0))
    left_shoulder_percentage = np.interp(body.left_shoulder_angle, (160, 175), (100, 0))
    if direction < 2:
        percentage = right_shoulder_percentage
    else:
        percentage = left_shoulder_percentage
    verify_rep_double_sided_exercises(percentage, direction)
