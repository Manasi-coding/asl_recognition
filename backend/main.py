from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import base64
import cv2
import numpy as np

from predictor import ASLPredictor

app = FastAPI()

# CORS (frontend talking to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ASLPredictor()

class ImageData(BaseModel):
    image: str

@app.post("/predict")
def predict(data: ImageData):
    try:
        # Decode base64 image
        header, encoded = data.image.split(",", 1)
        frame = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), np.uint8),
            cv2.IMREAD_COLOR
        )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        letter = predictor.process_frame(frame_rgb)

        return {
            "prediction": letter,
            "is_recording": predictor.recording,
            "gesture_type": predictor.gesture_type,
            "cooldown": predictor.cooldown,
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/classes")
def get_classes():
    """Return the model's class labels as a JSON list."""
    return {"classes": predictor.classes.tolist()}


@app.get("/api/config")
def get_config():
    """Return operational thresholds so the frontend can adapt dynamically."""
    return {
        "capture_interval_ms": 100,
        "movement_threshold": predictor.movement_threshold,
        "still_frames_threshold": predictor.still_frames_threshold,
        "cooldown_frames": predictor.cooldown_frames,
        "min_sequence_length": predictor.min_sequence_length,
        "labels": {
            "no_hand": "",
            "scanning": "\u2014",
        },
    }