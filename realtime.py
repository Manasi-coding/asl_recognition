import cv2 as cv
import mediapipe as mp
import numpy as np
import joblib
from features import extract_features_from_landmarks

# load the trained model and classes
model = joblib.load("best_model.pkl")
classes = np.load("classes.npy")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpdraw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # flip the frame horizontally for a mirror effect
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mpdraw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        features = extract_features_from_landmarks(hand.landmark)
        features = features.reshape(1, -1)

        pred = model.predict(features)[0]
        letter = classes[pred]

        cv.putText(
            frame,
            letter,
            (50, 100),
            cv.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            4
        )

    cv.imshow("ASL Alphabet Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()