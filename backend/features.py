import cv2 as cv
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands

def compute_angle(a, b, c):
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]

    dot = bax * bcx + bay * bcy

    mag_ba = math.sqrt(bax*bax + bay*bay)
    mag_bc = math.sqrt(bcx*bcx + bcy*bcy)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cos_angle = dot / (mag_ba * mag_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.acos(cos_angle)


def _normalize(landmarks):
    points = np.array([[lm.x, lm.y] for lm in landmarks])
    points -= points[0]
    palm = np.linalg.norm(points[9])
    if palm: points /= palm
    return points


def extract_features_from_landmarks(landmarks):
    points = _normalize(landmarks)
    
    features = [np.linalg.norm(points[i]) for i in [4, 8, 12, 16, 20]]

    for a, b, c in [
        (2,3,4),
        (5,6,8),
        (9,10,12),
        (13,14,16),
        (17,18,20)
        ]:
        
        ba, bc = points[a]-points[b], points[c]-points[b]
        mag = np.linalg.norm(ba) * np.linalg.norm(bc)
        cos = np.dot(ba, bc)/mag if mag else 0
        features.append(math.acos(max(-1.0, min(1.0, cos))))

    return np.array(features)


def extract_features_from_image(image_path, hands):
    image = cv.imread(image_path)

    if image is None:
        return None

    image = cv.resize(image, (224, 224))
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None

    landmarks = results.multi_hand_landmarks[0].landmark

    return extract_features_from_landmarks(landmarks)


def is_finger_extended(points, tip, pip):
    return points[tip][1] < points[pip][1]


def is_i_handshape(landmarks):
    p = _normalize(landmarks)
    return (is_finger_extended(p,20,18) and not is_finger_extended(p,8,6)
            and not is_finger_extended(p,12,10) and not is_finger_extended(p,16,14))


def is_d_handshape(landmarks):
    p = _normalize(landmarks)
    return (is_finger_extended(p,8,6) and not is_finger_extended(p,12,10)
            and not is_finger_extended(p,16,14) and not is_finger_extended(p,20,18))


def classify_sequence(sequence, gesture_type):
    if len(sequence) < 10 or gesture_type is None:
        return None

    ix = [p["index"][0] for p in sequence]
    px = [p["pinky"][0] for p in sequence]
    py = [p["pinky"][1] for p in sequence]

    if gesture_type == "J":
        if py[-1] > py[0] + 0.05 and px[-1] < px[0] - 0.05:
            return "J"

    elif gesture_type == "Z":
        segments, cur_dir, cur_len = [], None, 0

        for i in range(1, len(ix)):
            dx = ix[i] - ix[i-1]
            if abs(dx) < 0.003:
                continue
            d = 1 if dx > 0 else -1
            if d == cur_dir:
                cur_len += abs(dx)
            else:
                if cur_dir is not None and cur_len > 0.02:
                    segments.append(cur_len)
                cur_dir, cur_len = d, abs(dx)

        if cur_len > 0.02:
            segments.append(cur_len)

        if len(segments) >= 3 and (max(ix) - min(ix)) > 0.08:
            return "Z"

    return None