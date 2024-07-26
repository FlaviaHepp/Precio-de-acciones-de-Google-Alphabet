"""
Datos bursátiles de Google (Alphabet Inc) descargados de Yahoo Finance (GOOG): este conjunto de datos contiene datos bursátiles de Google 
(GOOG) para el año (de 2018 a 2024). Creo que los conocimientos extraídos de estos datos pueden utilizarse para crear algoritmos útiles 
de previsión de precios que ayuden a la inversión."""

#Importar bibliotecas
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

#Importar conjunto de datos
google_training_complete = pd.read_csv("GOOG.csv")
print(google_training_complete.head())

google_training_processed = google_training_complete.iloc[:, 1:2].values
print(google_training_processed)

scaler = MinMaxScaler(feature_range = (0, 1))
google_training_scaled = scaler.fit_transform(google_training_processed)

google_training_scaled.shape

features_set = []
labels = []

for i in range(60, google_training_scaled.shape[0]):
    features_set.append(google_training_scaled[i - 60:i, 0])
    labels.append(google_training_scaled[i, 0])
    
features_set, labels = np.array(features_set), np.array(labels)
features_set.shape

#Las capas LSTM funcionan con datos 3D con la siguiente estructura (nb_sequence, nb_timestep, nb_feature).

#nb_sequence corresponde al número total de secuencias en su conjunto de datos (o al tamaño del lote si está utilizando el aprendizaje por 
#mini 
# lotes).
#nb_timestep corresponde al tamaño de tus secuencias.
#nb_feature corresponde a la cantidad de funciones que describen cada uno de tus pasos de tiempo.
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
features_set.shape

#Modelo de entrenamiento
model = Sequential()

model.add(LSTM(units=300, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=100))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.summary()

model.fit(features_set, labels, epochs = 300, batch_size = 32)

#Importar conjunto de datos de prueba para predecir
google_testing_complete = pd.read_csv("GOOG-Jun-2024.csv")
google_testing_processed = google_testing_complete.iloc[:, 1:2].values
google_total = pd.concat((google_training_complete['Open'], google_testing_complete['Open']), axis=0)

test_inputs = google_total[len(google_total) - len(google_testing_complete) - 60:].values
test_inputs

test_inputs = test_inputs.reshape(-1, 1)
test_inputs = scaler.transform(test_inputs)
test_features = []

for i in range(60, 81):
    test_features.append(test_inputs[i - 60:i, 0])
test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
predictions = model.predict(test_features)

predictions = scaler.inverse_transform(predictions)

#Resultados de la trama
plt.style.use('dark_background')
plt.figure(figsize=(10,6))
plt.plot(google_testing_processed, color='blue', label='Precio real de las acciones de Apple', marker = "v")
plt.plot(predictions , color='red', label='Precio previsto de las acciones de Apple', marker = "v")
plt.title('Predicción del precio de las acciones de Apple\n', fontsize=16, fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Precio de las acciones de Apple\n')
plt.legend()
plt.show()

