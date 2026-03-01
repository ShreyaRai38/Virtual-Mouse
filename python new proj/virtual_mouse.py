import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# ---------------- Camera Setup ----------------
cam_width, cam_height = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

# ---------------- Hand Tracking Setup ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ---------------- Screen & Smoothening ----------------
screen_width, screen_height = pyautogui.size()
smoothening = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# ---------------- Click Delay ----------------
click_delay = 0.3
last_click_time = 0

# ---------------- Main Loop ----------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Mirror image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * cam_width), int(lm.y * cam_height)
                lm_list.append((id, cx, cy))

            if lm_list:
                # Get finger positions
                x_index, y_index = lm_list[8][1:]   # Index finger tip
                x_middle, y_middle = lm_list[12][1:]  # Middle finger tip
                x_thumb, y_thumb = lm_list[4][1:]   # Thumb tip

                # ---------------- Mouse Movement ----------------
                screen_x = np.interp(x_index, (0, cam_width), (0, screen_width))
                screen_y = np.interp(y_index, (0, cam_height), (0, screen_height))

                # Smoothen the movement
                curr_x = prev_x + (screen_x - prev_x) / smoothening
                curr_y = prev_y + (screen_y - prev_y) / smoothening
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # ---------------- Click Gesture ----------------
                distance = math.hypot(x_middle - x_index, y_middle - y_index)
                if distance < 30:
                    if time.time() - last_click_time > click_delay:
                        pyautogui.click()
                        last_click_time = time.time()

                # ---------------- Scroll Gesture ----------------
                if y_thumb < y_index - 40:
                    pyautogui.scroll(40)
                if y_thumb > y_index + 40:
                    pyautogui.scroll(-40)

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the webcam feed
    cv2.imshow("Virtual Mouse", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()