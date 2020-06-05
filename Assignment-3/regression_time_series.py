import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
import matplotlib.pyplot as plt
import math
import sys

df = pd.read_csv(sys.argv[1], sep=';', 
                 usecols=[2],na_values=['?','nan'], low_memory = False)
l = df[df['Global_active_power'].isnull()].index.tolist()

data = df.values
data = [j for sub in data for j in sub]
data = [i.tolist() for i in data]


trainX = list()
trainY = list()
for i in range(len(data)-60):
    data_check = np.isnan(data[i:i+61])
    if not np.sum(data_check):
        trainX.append(data[i:i+60])
        trainY.append(data[i+60])
trainX = np.array(trainX, dtype='float32')
trainY = np.array(trainY,  dtype='float32')

train_sample , test_sample = train_test_split(np.array(range(trainX.shape[0])))
testX,testY = trainX[test_sample],trainY[test_sample]
trainX, trainY = trainX[train_sample],trainY[train_sample]
testX.shape, trainX.shape, testY.shape, trainY.shape


look_back = 60
model = Sequential()
model.add(Dense(200, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='SGD')
model.fit(trainX, trainY, epochs=1, batch_size=128, verbose=2)
trainScore = model.evaluate(trainX, trainY, verbose=0)
testScore = model.evaluate(testX, testY, verbose=0)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

testPredict = testPredict.ravel()

data=np.array(data)
nan_vals = list()
for i in l:
    testX = data[i-60:i]
    data[i] = model(testX.reshape(1,-1))[0][0]
    nan_vals.append(data[i])
for i in nan_vals:
    print(i)