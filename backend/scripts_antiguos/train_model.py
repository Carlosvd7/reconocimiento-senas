import pandas as pd  # Para manejar datos en formato CSV
import tensorflow as tf  # Para crear y entrenar la red neuronal
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba
from sklearn.preprocessing import LabelEncoder  # Para convertir las etiquetas de texto en números
from tensorflow.keras.models import Sequential  # Para construir el modelo de red neuronal
from tensorflow.keras.layers import Dense, Dropout  # Para añadir capas densas (neuronas conectadas)

# Cargar los datos desde el archivo CSV
df = pd.read_csv("dataset/gestos_data.csv")  # Leer el dataset con los gestos

# Separar las características (X) y las etiquetas (y)
X = df.iloc[:, :-1].values  # Todas las columnas excepto la última (coordenadas de la mano)
y = df.iloc[:, -1].values   # Última columna (nombre del gesto)

# Convertir etiquetas de texto en números (Ejemplo: "Hola" → 0, "Adiós" → 1, etc.)
encoder = LabelEncoder()  # Inicializar el codificador
y = encoder.fit_transform(y)  # Convertir etiquetas de texto a números

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir la arquitectura del modelo de red neuronal
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),  # Antes era 128
    Dropout(0.2),  # Evita sobreajuste
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(set(y)), activation='softmax')  # Salida para cada letra
])

# Compilar el modelo
model.compile(optimizer='adam',  # Optimizador Adam (rápido y eficiente)
              loss='sparse_categorical_crossentropy',  # Función de pérdida para clasificación de múltiples clases
              metrics=['accuracy'])  # Métrica de evaluación: precisión del modelo

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train, 
          epochs=50,  # Número de veces que el modelo verá los datos (ajustable)
          validation_data=(X_test, y_test))  # Validación con datos de prueba

# Guardar el modelo entrenado para usarlo en la predicción
model.save("models/tensor_model.h5")  # Guardar en formato HDF5 (.h5) por que tensorflow usa formato HDF5
print("✅ Modelo entrenado y guardado correctamente.")
