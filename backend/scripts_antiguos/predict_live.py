import numpy as np  # Para manejar matrices numéricas
import tensorflow as tf  # Para cargar el modelo de IA
import cv2  # OpenCV para capturar video y procesar imágenes
import mediapipe as mp  # MediaPipe para la detección de manos
from sklearn.preprocessing import LabelEncoder  # Para decodificar etiquetas de gestos
import pandas as pd  # Para manejar datos en CSV
import requests  # Para enviar datos a la API Flask
import time  # Para controlar el tiempo entre envíos

#Cargar el modelo entrenado
model = tf.keras.models.load_model("../models/tensor_model.h5")
  # Cargar el modelo previamente entrenado

#Cargar etiquetas del entrenamiento
df = pd.read_csv("../dataset/gestos_data.csv")  # Leer las etiquetas guardadas en el dataset
encoder = LabelEncoder()  # Inicializar el codificador de etiquetas
encoder.fit(df.iloc[:, -1].values)  # Ajustar el codificador con las etiquetas de los gestos

#Inicializar MediaPipe Hands para detectar las manos en el video
mp_hands = mp.solutions.hands  # Cargar el módulo de detección de manos
hands = mp_hands.Hands()  # Inicializar el detector de manos con valores predeterminados
mp_draw = mp.solutions.drawing_utils  # Herramienta para dibujar los puntos clave en la mano

#Capturar video desde la cámara
cap = cv2.VideoCapture(0)  # Abrir la cámara principal (0)

last_sent_time = 0  # Última vez que se envió una letra
send_interval = 3  # Intervalo en segundos entre cada letra 
last_sent_letter = None  # Última letra enviada para evitar repeticiones

#Bucle para procesar el video en tiempo real
while cap.isOpened():
    ret, frame = cap.read()  # Capturar un frame de la cámara
    if not ret:  # Si no se pudo capturar el frame, salir del bucle
        break

    # Convertir la imagen a formato RGB (MediaPipe usa RGB, pero OpenCV usa BGR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)  # Procesar la imagen para detectar manos

    #Si se detecta una mano en la imagen
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar los puntos clave de la mano en la imagen
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer las coordenadas X, Y, Z de los 21 puntos clave de la mano
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Guardar las coordenadas en la lista

            #Convertir los datos en un array de NumPy para pasarlos al modelo
            input_data = np.array([landmarks])  # Convertir lista a matriz NumPy

            #Predecir el gesto usando el modelo entrenado
            prediction = model.predict(input_data)  # Hacer la predicción con la IA
            label = encoder.inverse_transform([np.argmax(prediction)])[0]  # Convertir el número a texto

            #Controlar el tiempo y evitar repetir la misma letra demasiado rápido
            current_time = time.time()  # Obtener el tiempo actual
            if (current_time - last_sent_time > send_interval) and (label != last_sent_letter):
                try:
                    requests.post("http://127.0.0.1:5001/update", json={"letter": label})  # Enviar la letra al servidor
                    last_sent_time = current_time  # Actualizar el tiempo de envío
                    last_sent_letter = label  # Guardar la última letra enviada para evitar repeticiones
                    print(f"✅ Letra enviada: {label}")  # Mostrar en consola la letra enviada
                except:
                    pass  # Si el servidor Flask no está corriendo, ignorar el error

            #Mostrar la letra en pantalla con OpenCV
            cv2.putText(frame, f"Gesto: {label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Texto en color rojo

    #Mostrar la imagen procesada en una ventana
    cv2.imshow("Reconocimiento de Gestos", frame)

    #Si el usuario presiona "q", salir del bucle y cerrar la ventana
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Liberar la cámara y cerrar las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
