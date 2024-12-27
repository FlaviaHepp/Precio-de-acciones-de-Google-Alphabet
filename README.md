# Precio-de-acciones-de-Google-Alphabet
Predicción de Precios de Acciones de Alphabet Inc. (Google) con Modelos LSTM

Se ingresó a Python y se visualizó el archivo Oracle Dataset.csv, utilizando la librería de Pandas. Este dataset, contiene información sobre precios de venta de acciones de la empresa NVIDIA del período 1986 al 2024. 
Se Realizó un análisis exploratorio de datos del Dataframe, especificando los principales hallazgos. 
Para desarrollar a profundidad el análisis se buscó dar respuestas a algunas inquietudes. Mediante la biblioteca de Plotly, se confeccionaron visualizaciones de datos con su correspondiente interpretación. Por último, se generó una pequeña conclusión del análisis efectuado.

Procesé datos financieros históricos de 2018 a 2024 descargados de Yahoo Finance, incluyendo escalado y generación de secuencias de tiempo con ventanas deslizantes.
Construí un modelo LSTM multicapa en Keras con regularización Dropout para prevenir el sobreajuste y optimizar el entrenamiento utilizando el algoritmo Adam.
Entrené el modelo durante 300 épocas y evalué su desempeño con la métrica de error cuadrático medio (MSE).
Implementé visualizaciones comparativas para analizar las predicciones del modelo frente a los precios reales, identificando tendencias clave.
Documenté las limitaciones del modelo y exploraré posibles mejoras, como la inclusión de datos externos relacionados con el mercado.
Habilidades adquiridas:
Modelado de series temporales con redes neuronales recurrentes (RNN) y LSTM.
Preprocesamiento avanzado de datos con pandas y MinMaxScaler.
Implementación de modelos de predicción con Keras y ajuste de hiperparámetros.
Visualización de resultados mediante matplotlib para análisis y comunicación de insights.
