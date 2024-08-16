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
    capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capture.set(cv2.CAP_PROP_FPS, 30)
    return capture


def start_video_capture(video_name):
    capture = cv2.VideoCapture(video_name)
    return capture


def build_body(frame, keypoints_with_scores):
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)

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


def get_body_and_display_frame(frame, keypoints, window):
    frame, body = build_body(frame, keypoints)

    # Uncomment to display angles
    Utils.display_angles(frame, body)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.rot90(frame_rgb)
    # frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    img = pygame.surfarray.make_surface(frame_rgb).convert()
    img = pygame.transform.flip(img, True, False)
    window.blit(img, (0, 0))

    # Update the display
    # pygame.display.flip()

    # Cap the frame rate
    clock.tick(Utils.fps)
    return body


def start_exercise(exercise, cap, keypoints, window):
    timer = time.time()
    remaining_reps = exercise.reps

    last_print_time = time.time()
    while time.time() - timer < exercise.elapsed_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        current_time = time.time()

        if current_time - last_print_time >= 1:
            print(f'Time remaining: {int(exercise.elapsed_time - (current_time - timer))}')
            last_print_time = current_time

        completed, direction = exercise.check_conditions()
        if completed:
            remaining_reps -= 1
            print(f'Reps remaining: {remaining_reps}')
            if remaining_reps == 0:
                print('Exercise complete!')
                return

        exercise.direction = direction
        exercise.body = get_body_and_display_frame(frame, keypoints, window)
        pygame.display.flip()


def main():
    pygame.init()
    window = pygame.display
    pygame.display.set_caption("MoveNet Lightning action recognition")

    cap = start_video_capture('ex2.mp4')

    if not cap.isOpened():
        print("Error: Could not open webcam/video.")
        return

    running = True

    try:
        while cap.isOpened() and running:
            ret, frame = cap.read()
            img = frame.copy()
            h, w, c = img.shape
            input_image = reshape_frame(img)
            keypoints_with_scores = get_key_points(input_image)
            # frame, body = build_body(frame, keypoints_with_scores)
            #
            # Utils.display_angles(frame, body)

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
                                False, True, 5, 30.0,
                                body,
                                conditions.left_elbow_bend_condition)
                    print(f'Starting exercise: {exercise.name}')
                    print(
                        f'You have {exercise.elapsed_time} seconds to complete {exercise.reps} reps')
                    start_exercise(exercise, cap, keypoints_with_scores, window.set_mode((w, h)))

                success, frame = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                body = get_body_and_display_frame(frame, keypoints_with_scores, window.set_mode((w,h)))
                pygame.display.flip()
            # cv2.imshow('MoveNet Lightning', frame)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    clock = pygame.time.Clock()
    main()
