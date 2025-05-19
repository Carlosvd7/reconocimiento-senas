# =============================
# video_to_sequence.py
# =============================
"""
Este script convierte un SOLO video (ej: gracias.mp4) en una secuencia .npy
con los 30 primeros frames procesados por MediaPipe Hands.
Ideal para pruebas individuales.
"""

import os
import cv2
import numpy as np
import mediapipe as mp

# === CONFIGURACION ===
video_path = "videos/gracias/gracias1.mp4"
output_folder = "dataset_sequences/gracias"
sequence_name = "gracias1"
SEQUENCE_LENGTH = 30

os.makedirs(output_folder, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(video_path)
sequence = []

while cap.isOpened() and len(sequence) < SEQUENCE_LENGTH:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    landmarks = np.zeros((21, 3))
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        for i, lm in enumerate(hand.landmark):
            landmarks[i] = [lm.x, lm.y, lm.z]

    sequence.append(landmarks.flatten())

cap.release()

while len(sequence) < SEQUENCE_LENGTH:
    sequence.append(sequence[-1])

sequence = np.array(sequence)
np.save(os.path.join(output_folder, f"{sequence_name}.npy"), sequence)
print(f"✅ Secuencia guardada en {output_folder}/{sequence_name}.npy")
