import cv2 as cv
import mediapipe as mp
import numpy as np
import joblib
import time
from features import extract_features_from_landmarks, classify_sequence, is_i_handshape, is_d_handshape

model   = joblib.load("best_model.pkl")
classes = np.load("classes.npy")
# scaler  = joblib.load("scaler.pkl")

hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
draw = mp.solutions.drawing_utils

prev_idx = prev_pky = None
sequence, recording = [], False
gesture_type, gesture_start = None, 0
still_frames, cooldown = 0, 0
display_letter = ""

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv.flip(frame, 1)
    results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

        idx = (hand.landmark[8].x, hand.landmark[8].y)
        pky = (hand.landmark[20].x, hand.landmark[20].y)

        movement = 0
        if prev_idx:
            movement = max(
                ((idx[0]-prev_idx[0])**2 + (idx[1]-prev_idx[1])**2)**0.5,
                ((pky[0]-prev_pky[0])**2 + (pky[1]-prev_pky[1])**2)**0.5
            )
        prev_idx, prev_pky = idx, pky

        if movement > 0.02 and not recording:
            recording = True
            still_frames = 0
            gesture_start = time.time()
            gesture_type = ("J" if is_i_handshape(hand.landmark)
                            else "Z" if is_d_handshape(hand.landmark)
                            else None)

        if recording:
            sequence.append({"index": idx, "pinky": pky})
            still_frames = still_frames + 1 if movement < 0.01 else 0

            if still_frames > 5 and time.time() - gesture_start > 0.8:
                recording = False
                cooldown = 5
                still_frames = 0
                if len(sequence) > 20:
                    result = classify_sequence(sequence, gesture_type)
                    if result: display_letter = result
                sequence = []

        elif cooldown > 0:
            cooldown -= 1
        else:
            features = extract_features_from_landmarks(hand.landmark).reshape(1, -1)
            pred = model.predict(features)[0]
            display_letter = classes[pred]

    else:
        sequence, recording = [], False
        prev_idx = prev_pky = None
        display_letter = ""

    cv.putText(frame, display_letter, (50, 100),
               cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
    cv.imshow("ASL Alphabet Recognition", frame)
    if cv.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv.destroyAllWindows()