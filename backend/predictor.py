import time
import numpy as np
import joblib
import mediapipe as mp
from collections import deque

from features import (
    extract_features_from_landmarks,
    classify_sequence,
    is_i_handshape,
    is_d_handshape
)

print("predictor.py is being imported")

class ASLPredictor:
    def __init__(self):
        self.model = joblib.load("model/best_model.pkl")
        self.classes = np.load("model/classes.npy")

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # State
        self.prev_idx = None
        self.prev_pky = None
        self.sequence = []
        self.recording = False
        self.gesture_type = None
        self.gesture_start = 0
        self.still_frames = 0
        self.cooldown = 0
        self.display_letter = ""

        # Tunable thresholds (read by /api/config)
        self.movement_threshold = 0.02
        self.still_threshold = 0.01
        self.still_frames_threshold = 5
        self.cooldown_frames = 5
        self.min_sequence_length = 20
        self.min_gesture_duration = 0.8

        # Smoothing buffer
        self.pred_buffer = deque(maxlen=3)

    def process_frame(self, frame):
        results = self.hands.process(frame)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            idx = (hand.landmark[8].x, hand.landmark[8].y)
            pky = (hand.landmark[20].x, hand.landmark[20].y)

            movement = 0
            if self.prev_idx:
                movement = max(
                    ((idx[0]-self.prev_idx[0])**2 + (idx[1]-self.prev_idx[1])**2)**0.5,
                    ((pky[0]-self.prev_pky[0])**2 + (pky[1]-self.prev_pky[1])**2)**0.5
                )

            self.prev_idx, self.prev_pky = idx, pky

            # Start dynamic gesture
            if movement > self.movement_threshold and not self.recording:
                self.recording = True
                self.still_frames = 0
                self.gesture_start = time.time()

                self.gesture_type = (
                    "J" if is_i_handshape(hand.landmark)
                    else "Z" if is_d_handshape(hand.landmark)
                    else None
                )

            if self.recording:
                self.sequence.append({"index": idx, "pinky": pky})
                self.still_frames = self.still_frames + 1 if movement < self.still_threshold else 0

                if self.still_frames > self.still_frames_threshold and time.time() - self.gesture_start > self.min_gesture_duration:
                    self.recording = False
                    self.cooldown = self.cooldown_frames
                    self.still_frames = 0
                    self.pred_buffer.clear()

                    if len(self.sequence) > self.min_sequence_length:
                        result = classify_sequence(self.sequence, self.gesture_type)
                        if result:
                            self.display_letter = result

                    self.sequence = []

            elif self.cooldown > 0:
                self.cooldown -= 1

            else:
                features = extract_features_from_landmarks(hand.landmark).reshape(1, -1)
                pred = self.model.predict(features)[0]
                label = self.classes[pred]

                self.pred_buffer.append(label)

                # Fast-confirm: if last 2 entries agree with new prediction, skip majority vote
                if len(self.pred_buffer) >= 2 and self.pred_buffer[-2] == label:
                    self.display_letter = label
                else:
                    self.display_letter = max(set(self.pred_buffer), key=self.pred_buffer.count)

        else:
            self.sequence = []
            self.recording = False
            self.prev_idx = None
            self.prev_pky = None
            self.pred_buffer.clear()
            self.display_letter = ""

        return self.display_letter