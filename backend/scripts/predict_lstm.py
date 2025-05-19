import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import requests
import time
import os
from sklearn.preprocessing import LabelEncoder

# Cargar modelo y etiquetas
model = tf.keras.models.load_model("models/lstm_model.h5")
labels = np.load("models/lstm_labels.npy", allow_pickle=True)
encoder = LabelEncoder()
encoder.classes_ = labels

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Captura desde webcam
cap = cv2.VideoCapture(0)

sequence = []
SEQUENCE_LENGTH = 30
last_sent_label = None
last_sent_time = 0
send_interval = 3  # segundos
current_label = ""  # para mostrar en pantalla

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer puntos
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            sequence.append(landmarks)

            if len(sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(sequence, axis=0)
                prediction = model.predict(input_data)
                label = encoder.inverse_transform([np.argmax(prediction)])[0]
                current_label = label  # actualizar texto mostrado
                current_time = time.time()

                # Verificar si el texto del backend está vacío
                try:
                    response = requests.get("http://localhost:5001/get_text")
                    backend_text = response.json().get("current_text", "")
                    if backend_text.strip() == "":
                        last_sent_label = None
                except:
                    pass

                if (current_time - last_sent_time > send_interval) and (label != last_sent_label):
                    try:
                        requests.post("http://localhost:5001/update_word", json={"word": label})
                        print(f"✅ Palabra enviada: {label}")
                        last_sent_label = label
                        last_sent_time = current_time
                    except:
                        print("Error al conectar con el servidor")

                sequence = []

    # Mostrar el gesto detectado en rojo
    if current_label:
        cv2.putText(frame, f"Gesto detectado: {current_label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Reconocimiento de Gestos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 