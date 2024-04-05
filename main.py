import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# %matplotlib inline

dataset_train = pd.read_csv("stock_price_train.csv")
# print(dataset_train.head())

training_set = dataset_train.iloc[:, 4:5].values

# print(training_set)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training_set = scaler.fit_transform(training_set)

# print(training_set.shape)

x_train = []
y_train = []

for i in range(60, 3056):
    x_train.append(scaled_training_set[i - 60:i, 0])
    y_train.append(scaled_training_set[i, 0])
x_train = np.array(x_train)
y_train = np.array(x_train)

# print(x_train.shape)
# print(y_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(x_train, y_train, epochs=10, batch_size=32)

dataset_test = pd.read_csv("stock_price_test.csv")
actual_stock_price = dataset_test.iloc[:, 4:5].values

dataset_total = pd.concat((dataset_train['close'], dataset_test['close']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

x_test = []

for i in range(60, 1000):
    x_test.append(inputs[i - 60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.plot(actual_stock_price, color='red', label='actual')
plt.plot(predicted_stock_price, color='blue', label='predicted')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()

plt.show()
