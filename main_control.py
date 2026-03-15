import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import time
import os
from mediapipe.tasks.python.vision import hand_landmarker

# --- CONFIGURATION ---
GESTURE_MODEL_PATH = "gesture_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
LANDMARKER_MODEL_PATH = "hand_landmarker.task"

pyautogui.FAILSAFE = True  # Move mouse to corner to stop

# 1. Load the AI Brain
model = joblib.load(GESTURE_MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# 2. Initialize Tasks API Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7
)
detector = HandLandmarker.create_from_options(options)

def normalize_landmarks(landmarks):
    temp_list = []
    base_x, base_y = landmarks[0].x, landmarks[0].y
    for lm in landmarks:
        temp_list.append([lm.x - base_x, lm.y - base_y])
    flat_list = np.array(temp_list).flatten()
    max_val = np.abs(flat_list).max()
    if max_val != 0: flat_list = flat_list / max_val
    return flat_list.reshape(1, -1)

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()
control_enabled = False

print("System Active! Use your 'Enable' gesture.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Tasks API requires timestamp in ms
    timestamp = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image, timestamp)

    if results.hand_landmarks:
        hand_lms = results.hand_landmarks[0]
        frame_h, frame_w, _ = frame.shape

        # Draw hand skeleton lines (Tasks API)
        for connection in hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS:
            start_lm = hand_lms[connection.start]
            end_lm = hand_lms[connection.end]

            x1, y1 = int(start_lm.x * frame_w), int(start_lm.y * frame_h)
            x2, y2 = int(end_lm.x * frame_w), int(end_lm.y * frame_h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Draw hand landmark dots
        for lm in hand_lms:
            px, py = int(lm.x * frame_w), int(lm.y * frame_h)
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
        
        # A. PREDICT GESTURE
        features = normalize_landmarks(hand_lms)
        prediction = model.predict(features)
        gesture_name = encoder.inverse_transform(prediction)[0]
        
        # B. GET CURSOR POSITION (Index tip is landmark 8)
        index_tip = hand_lms[8]
        # Mapping camera (0-1) to screen pixels
        cursor_x = np.interp(index_tip.x, [0, 1], [0, screen_w])
        cursor_y = np.interp(index_tip.y, [0, 1], [0, screen_h])

        # C. UI FEEDBACK
        status_color = (0, 255, 0) if control_enabled else (0, 0, 255)
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # D. ACTION LOGIC (Match these to your specific Labels)
        if gesture_name == "0": # e.g., OPEN_PALM
            control_enabled = True
        elif gesture_name == "1": # e.g., FIST
            control_enabled = False
        
        if control_enabled:
            if gesture_name == "2": # e.g., INDEX_UP
                pyautogui.moveTo(cursor_x, cursor_y, _pause=False)
            elif gesture_name == "3": # e.g., PINCH
                pyautogui.click()
                time.sleep(0.3)
            elif gesture_name == "4": # e.g., TWO_FINGERS
                pyautogui.scroll(20)
            elif gesture_name == "5": # e.g., POINT_LEFT
                pyautogui.press('left')
                time.sleep(0.4)
            elif gesture_name == "6": # e.g., POINT_RIGHT
                pyautogui.press('right')
                time.sleep(0.4)

    cv2.imshow("Tasks API Control System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
detector.close()