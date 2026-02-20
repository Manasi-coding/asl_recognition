import cv2 as cv
import mediapipe as mp
import numpy as np
import math

def extract_features (landmarks):
    points = np.array([[lm.x, lm.y] for lm in landmarks])

    # 1. make the wrist's co-ordinate (0,0)
    wrist = points[0]
    points = points - wrist

    # 2. use relative distancing
    palm_size = math.sqrt(points[9][0]**2 + points[9][1]**2)
    if palm_size != 0 :
        points = points / palm_size
    
    features = []

    # 3. distances between the fingertips
    fingertips = [4, 8, 12, 16, 20]
    for i in fingertips:
        distance = math.sqrt(points[i][0]**2 + points[i][1]**2)
        features.append(distance)

    # 4. angles 
    def angle (a, b, c):
        bax = a[0] - b[0]
        bay = a[1] - b[1]
        bcx = c[0] - b[0]
        bcy = c[1] - b[1]

        # dot product
        dot = bax * bcx + bay * bcy

        # magnitudes
        mag_ba = math.sqrt(bax*bax + bay*bay)
        mag_bc = math.sqrt(bcx*bcx + bcy*bcy)

        # cosine of angle
        cos_angle = dot / (mag_ba * mag_bc)

        # angle must be between -1 and 1
        cos_angle = max(-1.0, min(1.0, cos_angle))

        return math.acos(cos_angle)
    
    finger_joints = [
        (2, 3, 4),
        (5, 6, 8),
        (9, 10, 12),
        (13, 14, 16),
        (17, 18, 20)
    ]

    for a, b, c in finger_joints:
        features.append(angle(points[a], points[b], points[c]))

    return np.array(features)