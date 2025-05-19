import os
import cv2
import numpy as np
import mediapipe as mp

# Configuración
SEQUENCE_LENGTH = 30  # Número de frames por secuencia
DATA_PATH = "dataset_sequences"  # Carpeta base

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Preguntar nombre del gesto/palabra
label = input("Introduce la palabra o gesto a grabar: ")
label_path = os.path.join(DATA_PATH, label)
os.makedirs(label_path, exist_ok=True)

# Iniciar cámara
cap = cv2.VideoCapture(0)
print(f"Grabando gestos para: {label}")
print("Presiona ESPACIO para grabar una secuencia de 30 frames. ESC para salir.")

sample_count = len(os.listdir(label_path))  # Contador de muestras ya guardadas

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Gesto: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "ESPACIO = grabar, ESC = salir", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Captura de secuencias", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # ESPACIO
        print("⏺️ Grabando secuencia...")
        sequence = []

        for _ in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            landmarks = np.zeros((21, 3))  # 21 puntos (x, y, z)
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                for i, lm in enumerate(hand.landmark):
                    landmarks[i] = [lm.x, lm.y, lm.z]

            sequence.append(landmarks.flatten())
            cv2.putText(frame, "Grabando...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Captura de secuencias", frame)
            cv2.waitKey(30)

        sequence = np.array(sequence)
        filename = os.path.join(label_path, f"{sample_count}.npy")
        np.save(filename, sequence)
        print(f"✅ Secuencia guardada como: {filename}")
        sample_count += 1

cap.release()
cv2.destroyAllWindows()
