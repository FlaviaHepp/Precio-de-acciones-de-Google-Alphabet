# 📈Predicción del Precio de las Acciones de Google Alphabet usando LSTM

Este proyecto implementa un modelo de Deep Learning basado en redes LSTM (Long Short-Term Memory) para predecir el precio de las acciones de Google Alphabet (GOOG) a partir de datos históricos bursátiles.

El enfoque está orientado al modelado de series temporales financieras, capturando dependencias de largo plazo en los precios.

# 🎯Objetivo del proyecto

- Modelar la evolución temporal del precio de las acciones de Google.
- Preparar los datos financieros para redes neuronales recurrentes.
- Entrenar una red LSTM profunda para predicción de precios.
- Evaluar visualmente la capacidad predictiva del modelo sobre datos no vistos.
- Explorar el uso de Deep Learning en mercados financieros.

# 📁Descripción de los datos

Se utilizan dos conjuntos de datos:
- Datos de entrenamiento
GOOG.csv

- Precio de apertura histórico de Google Alphabet.
  
- Datos de prueba
GOOG-Jun-2024.csv

- Precios reales utilizados para validar el modelo.
  
- Variable utilizada:
  -- Open: precio de apertura de la acción.

# 🔄Preprocesamiento de datos

- Selección del precio de apertura.
- Normalización con MinMaxScaler (0–1).
- Creación de ventanas temporales de 60 días.

- Transformación de los datos a formato 3D requerido por LSTM:
  -- (n_samples, timesteps, n_features)

# 🧠Arquitectura del modelo

Red neuronal profunda basada en LSTM:
- 4 capas LSTM (300 → 100 → 100 → 100 unidades)
- Capas Dropout (0.2) para regularización
- Capa densa final para predicción del precio
- Función de pérdida: Mean Squared Error
- Optimizador: Adam
  
- Entrenamiento:
  -- 300 epochs
  -- Batch size: 32

# 📊Resultados

- Comparación gráfica entre:
- Precio real de las acciones
- Precio predicho por el modelo LSTM
- El modelo captura correctamente la tendencia general del precio.
- Se observa la capacidad de la red para modelar dependencias temporales de largo plazo.

# 🛠️Tecnologías utilizadas

- Python
- pandas / numpy
- Matplotlib
- scikit-learn
- Keras / TensorFlow
- Deep Learning (LSTM)

# 📂Estructura del proyecto

├── Predicción de precios de acciones Google Alphabet.py
├── GOOG.csv
├── GOOG-Jun-2024.csv
└── README.md


# 📌Consideraciones

- El modelo utiliza solo precios históricos, sin variables macroeconómicas.
- No se realiza optimización de hiperparámetros.
- La predicción es sensible a la ventana temporal seleccionada.
- Ideal como demostración de Deep Learning aplicado a finanzas.

# ⚠️Disclaimer

Este proyecto tiene fines educativos y demostrativos.
No constituye asesoramiento financiero ni recomendaciones de inversión.

# 👤Autor

Flavia Hepp
Data Science en formación· Deep Learning · Series Temporales
