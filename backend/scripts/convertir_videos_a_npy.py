# =============================
# convertir_videos_a_npy.py
# =============================
"""
Este script recorre TODAS las carpetas dentro de "videos/" y convierte cada
archivo de video (.mp4, .mov, .avi) en una secuencia .npy válida para LSTM.
Ideal para construir el dataset completo de forma automática.
"""

import os
import cv2
import numpy as np
import mediapipe as mp

VIDEOS_BASE_PATH = "videos"
OUTPUT_BASE_PATH = "dataset_sequences"
SEQUENCE_LENGTH = 30

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

for label in os.listdir(VIDEOS_BASE_PATH):
    label_video_path = os.path.join(VIDEOS_BASE_PATH, label)
    if not os.path.isdir(label_video_path):
        continue

    output_label_path = os.path.join(OUTPUT_BASE_PATH, label)
    os.makedirs(output_label_path, exist_ok=True)

    for filename in os.listdir(label_video_path):
        if filename.endswith(('.mp4', '.mov', '.avi')):
            video_path = os.path.join(label_video_path, filename)
            cap = cv2.VideoCapture(video_path)
            sequence = []
            frame_count = 0

            while cap.isOpened() and frame_count < SEQUENCE_LENGTH:
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
                frame_count += 1

            cap.release()

            while len(sequence) < SEQUENCE_LENGTH:
                sequence.append(sequence[-1])

            sequence = np.array(sequence)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_label_path, f"{base_name}.npy")
            np.save(output_path, sequence)
            print(f"✅ Guardado: {output_path}")

print("✅ Conversión automática de vídeos completada.")