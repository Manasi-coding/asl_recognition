import cv2 as cv
import mediapipe as mp
from features import extract_features
from model import model

mphands = mp.solutions.hands
hands = mphands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) # take only one hand
mpdraw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # mirror view
    frame = cv.flip(frame, 1)

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mpdraw.draw_landmarks(frame, hand, mphands.HAND_CONNECTIONS)

        features = extract_features(hand.landmark)
        features = features.reshape(1, -1)

        pred = model.predict(features)[0]

        if pred == 0:
            letter = "A"
        elif pred == 1:
            letter = "B"
        elif pred == 2:
            letter = "C"
        else:
            letter = "?"

        cv.putText(
            frame,
            letter,
            (50, 100),
            cv.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            4
        )

    cv.imshow("ASL Recognition", frame)

    # exit on ESC (27)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()