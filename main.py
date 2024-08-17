import time
import pygame
import conditions
import tensorflow as tf
import numpy as np
import cv2

from utils import Utils
from body import Body
from sprite import AnimatedSprite
from utils import Utils
from exercise import Exercise

interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()
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


def start_webcam_capture():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    # capture.set(cv2.CAP_PROP_FPS, 30)
    return capture


def start_video_capture(video_name):
    capture = cv2.VideoCapture(video_name)
    return capture


def build_frame_and_body(cap):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Ignoring empty camera frame.")

    img = frame.copy()
    input_image = reshape_frame(img)
    keypoints_with_scores = get_key_points(input_image)

    draw_connections(frame, keypoints_with_scores, EDGES, 0.3)
    draw_keypoints(frame, keypoints_with_scores, 0.3)

    keypoints_array = keypoints_with_scores[0][0]
    body = Body(**{name: keypoints_array[landmark]
                   for name, landmark in Utils.landmark_dict.items()})
    return frame, body


def draw_keypoints(frame, keypoints, confidence):
    y, x, z = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_treshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_treshold) & (c2 > confidence_treshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)


def reshape_frame(img):
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)
    return input_image


def get_key_points(input_image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores


def draw_get_in_frame_border(window, is_standing):
    if window is None:
        raise ValueError("Window is None")

    # Standing position frame
    standing_rect_origin_x, standing_rect_origin_y = 240, 50
    standing_rect_width, standing_rect_height = 800, 924

    # Laying position frame
    laying_rect_origin_x, laying_rect_origin_y = 50, 150
    laying_rect_width, laying_rect_height = 1180, 724

    if is_standing:
        standing_rect_sizes = (standing_rect_origin_x, standing_rect_origin_y,
                               standing_rect_width, standing_rect_height)
        pygame.draw.rect(window, (221, 221, 221, 70), standing_rect_sizes, 4,
                         border_radius=20)
    else:
        laying_rect_sizes = (laying_rect_origin_x, laying_rect_origin_y,
                             laying_rect_width, laying_rect_height)
        pygame.draw.rect(window, (221, 221, 221, 70), laying_rect_sizes, 4,
                         border_radius=20)

    return window


def check_if_body_in_frame(body, is_standing):
    if body is None:
        return False
    # print(f' Nose: {body.nose}')
    # print(f' Right heel: {body.right_heel}')
    # print(f' Left heel: {body.left_heel}')
    if is_standing:
        if (0.4 < body.nose[0] < 0.6 and 0.05 < body.nose[1] < 0.2
                and 0.4 < body.right_heel[0] < 0.6 and 0.8 < body.right_heel[1] < 0.9
                and 0.4 < body.left_heel[0] < 0.7 and 0.8 < body.left_heel[1] < 0.9):
            return True
    else:
        if (0.1 < body.nose[0] < 0.3 and 0.2 < body.nose[1] < 0.5
                and 0.6 < body.right_heel[0] < 0.75
                and 0.6 < body.right_heel[1] < 0.8):
            return True

    return False


def wait_for_body_in_frame(cap, window, is_standing):
    countdown = 3
    last_decrement_time = time.time()

    while countdown > -1:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                return False

        body = get_body_and_display_frame(cap, window)
        draw_get_in_frame_border(window, is_standing)

        # Update the display
        pygame.display.flip()

        # Start countdown if body is in frame
        current_time = time.time()
        is_body_in_frame = check_if_body_in_frame(body, is_standing)
        if is_body_in_frame and current_time - last_decrement_time >= 1:
            countdown -= 1
            last_decrement_time = current_time
            print(f'Countdown: {countdown}')
        elif not is_body_in_frame:
            countdown = 3

    return True


def get_body_and_display_frame(cap, window):
    frame, body = build_frame_and_body(cap)
    # print(body.)
    # Uncomment to display angles
    Utils.display_angles(frame, body)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.rot90(frame_rgb)

    img = pygame.surfarray.make_surface(frame_rgb).convert()
    img = pygame.transform.flip(img, True, False)
    img = pygame.transform.scale(img, (1280, 1024))
    window.blit(img, (0, 0))

    # Cap the frame rate
    clock.tick(Utils.fps)
    return body


def start_exercise(exercise, cap, window):
    timer = time.time()
    remaining_reps = exercise.reps
    animation_frame_list = AnimatedSprite.loadGIF(exercise.image_url)
    animated_sprite = AnimatedSprite(75, Utils.height, animation_frame_list)
    all_sprites = pygame.sprite.Group(animated_sprite)

    last_print_time = time.time()
    while time.time() - timer < exercise.elapsed_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()

        current_time = time.time()

        if current_time - last_print_time >= 1:
            print(f'Time remaining: {int(exercise.elapsed_time - (current_time - timer))}')
            last_print_time = current_time

        exercise.body = get_body_and_display_frame(cap, window)
        completed, direction = exercise.check_conditions()
        if completed:
            remaining_reps -= 1
            print(f'Reps remaining: {remaining_reps}')
            if remaining_reps == 0:
                print('Exercise complete!')
                return

        exercise.direction = direction

        all_sprites.update()
        all_sprites.draw(window)
        pygame.display.flip()
    print('Time up!')
    return


def main():
    pygame.init()
    window = pygame.display.set_mode((1280, 1024))
    pygame.display.set_caption("MoveNet Lightning action recognition")

    # cap = start_video_capture('ex2.mp4')
    cap = start_webcam_capture()

    if not cap.isOpened():
        print("Error: Could not open webcam/video.")
        return

    running = True

    try:
        while cap.isOpened() and running:
            body = get_body_and_display_frame(cap, window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False
                    pygame.quit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    print('Loading exercise...')
                    exercise = Exercise("Elbow bends", "./Resources/Woman doing Side Lunges.gif",
                                        False, True, True,
                                        5, 30.0,
                                        body,
                                        conditions.left_elbow_bend)
                    if not wait_for_body_in_frame(cap, window, exercise.is_standing):
                        continue
                    print(f'Starting exercise: {exercise.name}')
                    print(f'You have {exercise.elapsed_time} seconds to complete {exercise.reps} reps')
                    start_exercise(exercise, cap, window)

            pygame.display.flip()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    clock = pygame.time.Clock()
    main()
