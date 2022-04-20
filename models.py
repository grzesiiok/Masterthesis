import keras.models
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import downloaddata
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from tensorflow.keras import Sequential,utils
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout, LSTM

def data(name):
    X = downloaddata.firm_data(name).minmax()[0]
    y = downloaddata.firm_data(name).minmax()[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test

def model_regression(X_train, X_test, y_train, y_test):
    Acc = []
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_1 = model.predict(X_test)
    pred_df = pd.DataFrame({'Rzeczywiste': y_test, 'Przewidziane': y_pred_1})
    print(pred_df)
    print("Dokładność prognozy: {0}".format(r2_score(y_test, y_pred_1)))
    Acc.append(r2_score(y_test, y_pred_1))
    plt.figure(figsize=(8, 8))
    plt.ylabel('Bilans zamknięcia', fontsize=16)
    plt.plot(pred_df)
    plt.legend(['Rzeczywiste', 'Przewidziane'])
    plt.show()

def ANN(X_train, y_train):
    model = Sequential()

    model.add(Dense(20, input_dim=6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(70, kernel_initializer='normal', activation='relu'))#
    model.add(Dense(70, kernel_initializer='normal', activation='relu'))#
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=70, validation_split=0.2)
    model.save('modelANN.h5')

def use_model(model_name,Ticker):
    dane = data(Ticker)
    model = keras.models.load_model(model_name)
    if (model_name == 'modelCNN.h5'):
        X_test = np.array(dane[1]).reshape(dane[1].shape[0], dane[1].shape[1], 1)
    elif(model_name == 'modelLSTM.h5'):
        X_test = dane[1].reshape(dane[1].shape[0], dane[1].shape[1], 1)
    else:
        X_test = dane[1]
    y_pred_2 = model.predict(X_test)
    pred_df = pd.DataFrame({'Rzeczywiste': dane[3], 'Przewidziane': y_pred_2.flatten()})
    print(pred_df.head())
    print("Dokładność prognozy: {0}".format(r2_score(dane[3], y_pred_2)))
    plt.figure(figsize=(8,8))
    plt.ylabel('Bilans zamknięcia', fontsize=16)
    plt.plot(pred_df)
    plt.legend(['Rzeczywiste', 'Przewidziane'])
    plt.show()

def CNN(X_train, y_train):
    X_train = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)

    model = Sequential()

    model.add(Conv1D(32, kernel_size=(3,), padding='same', activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Conv1D(64, kernel_size=(3,), padding='same', activation='relu'))
    model.add(Conv1D(128, kernel_size=(5,), padding='same', activation='relu'))

    model.add(Flatten())

    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=60, validation_split=0.2)
    model.save('modelCNN.h5')

def LSTM1(X_train, y_train):
    X_train_ = X_train.values.reshape(X_train.shape[0],X_train.shape[1],1)
    # print(X_train_)
    #X_train_ = X_train

    model = Sequential()

    model.add(LSTM(70, return_sequences=True, input_shape=(30, 6)))
    model.add(LSTM(70, return_sequences=True))
    model.add(LSTM(70))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train_, y_train, epochs=100, validation_split=0.2)

    model.save('modelLSTM.h5')

dane = data('AAPL')
model_regression(dane[0], dane[1],dane[2], dane[3])
#LSTM1(dane[0], dane[2])
#use_model('modelLSTM.h5', 'AAPL')