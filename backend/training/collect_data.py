import os 
import cv2 as cv
import mediapipe as mp
import numpy as np
from features import extract_features

mphands = mp.solutions.hands
hands = mphands.Hands(static_image_mode=True) # looking at the images

X = []
y = []

label_map = {"A": 0, "B": 1, "C": 2}

for label in ["A", "B", "C"]:
    folder = f"dataset/{label}"

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv.imread(img_path)

        if img is None:
            continue

        rgb_frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            features = extract_features(hand.landmark)

            X.append(features)
            y.append(label_map[label])

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)

print("Saved dataset:", X.shape, y.shape)