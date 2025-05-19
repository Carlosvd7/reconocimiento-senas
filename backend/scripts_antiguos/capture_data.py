import os
import cv2
import mediapipe as mp
import csv

#Definir la ruta correcta del archivo CSV
csv_file = "dataset/gestos_data.csv"

#Asegurar que la carpeta dataset/ existe antes de abrir el archivo
if not os.path.exists("dataset"):
    os.makedirs("dataset")  # Si no existe, la crea automáticamente

#Crear el archivo CSV si no existe o si está vacío
if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x" + str(i+1) for i in range(21*3)] + ["Etiqueta"])  # Encabezados de coordenadas y etiqueta

# Inicializar MediaPipe para la detección de manos
mp_hands = mp.solutions.hands  # Cargar el módulo de detección de manos de MediaPipe
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)  
# El detector de manos detectará si hay al menos un 50% de certeza y solo rastreará si está seguro en un 50%.

mp_draw = mp.solutions.drawing_utils  # Herramienta para dibujar los puntos de la mano

#Pedir al usuario el nombre del gesto antes de empezar a capturar
gesture_label = input("Ingresa el nombre del gesto (Ejemplo: A, B, hola, adiós): ")

#Capturar video desde la cámara
cap = cv2.VideoCapture(0)
print("Mostrando cámara... Realiza el gesto y presiona 's' para guardar datos.")

#Abrir el archivo CSV en modo "append" para agregar datos sin sobrescribir los anteriores
with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)

    while cap.isOpened():  # Mientras la cámara esté abierta
        ret, frame = cap.read()  # Capturar un fotograma (frame) de la cámara
        
        if not ret:  # Si la cámara no devuelve un frame válido, salir del bucle
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir a formato RGB para MediaPipe
        results = hands.process(frame_rgb)  # Procesar la imagen para detectar manos

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Dibujar los puntos y conexiones en la imagen de la cámara

                #Extraer coordenadas X, Y, Z de los 21 puntos clave de la mano
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])  

                #Mostrar mensaje en pantalla para el usuario
                cv2.putText(frame, "Presiona 's' para guardar", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                #Guardar datos en el archivo CSV al presionar "s"
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    landmarks.append(gesture_label)  # Agregar la etiqueta del gesto
                    writer.writerow(landmarks)  # Guardar en el CSV
                    print(f"✅ Datos guardados para el gesto: {gesture_label}")

        cv2.imshow("Captura de Gestos", frame)  # Mostrar la imagen con los puntos de la mano dibujados

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Si el usuario presiona "q", salir del bucle y cerrar la cámara.

cap.release()  # Cierra la cámara
cv2.destroyAllWindows()  # Cierra la ventana de OpenCV.



from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)




