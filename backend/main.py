import asyncio
import time
import pathlib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Explicit import when running from root directory
from backend.model import DrowsinessPredictor

# 1. Initialize FastAPI app (Uvicorn looks for this exact name)
app = FastAPI(title="EEG Driver Drowsiness Detection Stream")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Resolve relative paths to modelfiles directory
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
WEIGHTS_PATH = BASE_DIR / "modelfiles" / "interpretable_cnn_weights.pt"
DEMO_X_PATH = BASE_DIR / "modelfiles" / "demo_subject_x.npy"
DEMO_Y_PATH = BASE_DIR / "modelfiles" / "demo_subject_y.npy"

# Load predictor and data
predictor = DrowsinessPredictor(str(WEIGHTS_PATH))
demo_x = np.load(str(DEMO_X_PATH))  # Shape: (N, 1, 30, 384)
demo_y = np.load(str(DEMO_Y_PATH))  # Shape: (N,)

# Selected channels for UI dashboard
DISPLAY_CHANNEL_INDICES = [4, 14, 24, 28] 
DISPLAY_CHANNEL_NAMES = ["Fz", "Cz", "Pz", "Oz"]

@app.websocket("/ws/eeg-stream")
async def websocket_eeg_stream(websocket: WebSocket):
    await websocket.accept()
    
    consecutive_drowsy_count = 0
    sample_index = 0
    total_samples = len(demo_x)

    try:
        while True:
            epoch = demo_x[sample_index, 0]  # Shape: (30, 384)
            true_label = int(demo_y[sample_index])

            # Model inference
            drowsy_prob = predictor.predict_epoch(epoch)

            # Backend Rolling Alert Logic (>75% probability for 3 consecutive epochs)
            if drowsy_prob >= 0.75:
                consecutive_drowsy_count += 1
            else:
                consecutive_drowsy_count = 0

            alert_active = consecutive_drowsy_count >= 3

            # Channel payload
            channels_payload = {}
            for idx, name in zip(DISPLAY_CHANNEL_INDICES, DISPLAY_CHANNEL_NAMES):
                channels_payload[name] = epoch[idx].tolist()

            payload = {
                "timestamp": time.time(),
                "epoch_index": sample_index,
                "drowsy_probability": round(drowsy_prob, 4),
                "ground_truth": true_label,
                "consecutive_drowsy_epochs": consecutive_drowsy_count,
                "alert": alert_active,
                "channels": channels_payload
            }

            await websocket.send_json(payload)

            sample_index = (sample_index + 1) % total_samples
            await asyncio.sleep(3.0)

    except WebSocketDisconnect:
        print("Client disconnected from EEG stream.")