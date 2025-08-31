import cv2
import mediapipe as mp
import pyautogui
import math
import time

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

click_cooldown = 0.5  # seconds
last_click_time = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            # Get index finger tip (id=8) and thumb tip (id=4)
            index = landmarks[8]
            thumb = landmarks[4]

            index_x = int(index.x * frame_width)
            index_y = int(index.y * frame_height)
            thumb_x = int(thumb.x * frame_width)
            thumb_y = int(thumb.y * frame_height)

            # Draw circles
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 0, 255), -1)

            # Move cursor with index finger
            screen_x = int(index.x * screen_width)
            screen_y = int(index.y * screen_height)
            pyautogui.moveTo(screen_x, screen_y)

            # Calculate distance between index and thumb
            distance = math.hypot(thumb_x - index_x, thumb_y - index_y)

            # If fingers close, perform click
            if distance < 40:  # threshold
                current_time = time.time()
                if current_time - last_click_time > click_cooldown:
                    pyautogui.click()
                    last_click_time = current_time

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
