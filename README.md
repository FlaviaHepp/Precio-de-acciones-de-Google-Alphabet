# Precio-de-acciones-de-Google-Alphabet
Predicción de Precios de Acciones de Alphabet Inc. (Google) con Modelos LSTM

Los datos bursátiles de Google (Alphabet Inc) se descargaron de Yahoo Finance (GOOG) teniendo como fecha de referencia 2018 a 2024. 

**Desarrollo:**
Procesé datos financieros históricos de 2018 a 2024 descargados de Yahoo Finance, incluyendo escalado y generación de secuencias de tiempo con ventanas deslizantes.
Construí un modelo LSTM multicapa en Keras con regularización Dropout para prevenir el sobreajuste y optimizar el entrenamiento utilizando el algoritmo Adam.
Entrené el modelo durante 300 épocas y evalué su desempeño con la métrica de error cuadrático medio (MSE).
Implementé visualizaciones comparativas para analizar las predicciones del modelo frente a los precios reales, identificando tendencias clave.
Documenté las limitaciones del modelo y exploraré posibles mejoras, como la inclusión de datos externos relacionados con el mercado.
*Herramientas utilizadas:* Python, pandas, numpy, matplotlib, keras y scikit-learn.
*Habilidades adquiridas:*
Modelado de series temporales con redes neuronales recurrentes (RNN) y LSTM.
Preprocesamiento avanzado de datos con pandas y MinMaxScaler.
Implementación de modelos de predicción con Keras y ajuste de hiperparámetros.
Visualización de resultados mediante matplotlib para análisis y comunicación de insights. 

📈 Predicción del Precio de las Acciones de Google Alphabet usando LSTM

Este proyecto implementa un modelo de Deep Learning basado en redes LSTM (Long Short-Term Memory) para predecir el precio de las acciones de Google Alphabet (GOOG) a partir de datos históricos bursátiles.

El enfoque está orientado al modelado de series temporales financieras, capturando dependencias de largo plazo en los precios.

🎯 Objetivo del proyecto

Modelar la evolución temporal del precio de las acciones de Google.

Preparar los datos financieros para redes neuronales recurrentes.

Entrenar una red LSTM profunda para predicción de precios.

Evaluar visualmente la capacidad predictiva del modelo sobre datos no vistos.

Explorar el uso de Deep Learning en mercados financieros.

📁 Descripción de los datos

Se utilizan dos conjuntos de datos:

Datos de entrenamiento

GOOG.csv

Precio de apertura histórico de Google Alphabet.

Datos de prueba

GOOG-Jun-2024.csv

Precios reales utilizados para validar el modelo.

Variable utilizada:

Open: precio de apertura de la acción.

🔄 Preprocesamiento de datos

Selección del precio de apertura.

Normalización con MinMaxScaler (0–1).

Creación de ventanas temporales de 60 días.

Transformación de los datos a formato 3D requerido por LSTM:

(n_samples, timesteps, n_features)

🧠 Arquitectura del modelo

Red neuronal profunda basada en LSTM:

4 capas LSTM (300 → 100 → 100 → 100 unidades)

Capas Dropout (0.2) para regularización

Capa densa final para predicción del precio

Función de pérdida: Mean Squared Error

Optimizador: Adam

Entrenamiento:

300 epochs

Batch size: 32

📊 Resultados

Comparación gráfica entre:

Precio real de las acciones

Precio predicho por el modelo LSTM

El modelo captura correctamente la tendencia general del precio.

Se observa la capacidad de la red para modelar dependencias temporales de largo plazo.

🛠️ Tecnologías utilizadas

Python

pandas / numpy

Matplotlib

scikit-learn

Keras / TensorFlow

Deep Learning (LSTM)

📂 Estructura del proyecto
├── Predicción de precios de acciones Google Alphabet.py
├── GOOG.csv
├── GOOG-Jun-2024.csv
└── README.md

▶️ Cómo ejecutar el proyecto

Clonar el repositorio

git clone https://github.com/tu_usuario/nombre_del_repo.git


Instalar dependencias

pip install pandas numpy matplotlib scikit-learn tensorflow keras


Ejecutar el script

python "Predicción de precios de acciones Google Alphabet.py"

📌 Consideraciones

El modelo utiliza solo precios históricos, sin variables macroeconómicas.

No se realiza optimización de hiperparámetros.

La predicción es sensible a la ventana temporal seleccionada.

Ideal como demostración de Deep Learning aplicado a finanzas.

⚠️ Disclaimer

Este proyecto tiene fines educativos y demostrativos.
No constituye asesoramiento financiero ni recomendaciones de inversión.

👤 Autor

Flavia Hepp
Data Science · Deep Learning · Series Temporales
