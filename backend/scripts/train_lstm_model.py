import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Configuración
DATASET_PATH = "dataset_sequences"
SEQUENCE_LENGTH = 30

sequences = []
labels = []

# Recorrer recursivamente
for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".npy"):
            path = os.path.join(root, file)
            sequence = np.load(path)
            if sequence.shape == (SEQUENCE_LENGTH, 63):
                sequences.append(sequence)
                label = os.path.basename(os.path.dirname(path))
                labels.append(label)
                print(f"Cargado: {path}")
            else:
                print(f"⚠️ Ignorado (forma inválida): {path} — {sequence.shape}")

print(f"1.Total de secuencias: {len(sequences)}")
print(f"2.Clases detectadas: {sorted(set(labels))}")

# Validación
if len(sequences) == 0:
    raise ValueError(" No se encontraron secuencias válidas. Verifica las rutas y los .npy.")

# Preparación
X = np.array(sequences)
y = np.array(labels)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Modelo
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 63)),
    Dropout(0.3),
    LSTM(64, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Guardado
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.h5")
np.save("models/lstm_labels.npy", encoder.classes_)
print("Modelo y etiquetas guardados en /models")
