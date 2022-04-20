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
import yfinance as yf
from tensorflow.keras.optimizers import SGD



def data(name):
    X = downloaddata.firm_data(name).minmax()[0]
    y = downloaddata.firm_data(name).minmax()[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test

def data2(name):
    df = yf.Ticker(name)
    history = df.history(period='max', interval='1d')
    # print(history)
    history["NonEmpty"] = 0.0
    idx = pd.date_range(history.index[0], history.index[-1])
    h = history.reindex(idx, fill_value=-1)
    # print(h)
    M = h.to_numpy(dtype=np.float32)
    # print(M.shape)
    # print(M.dtype)
    M[:, 4] *= (1 / 500000000)
    X0 = M[1:-31, :]
    X1 = M[1 + 1:-31 + 1, :]  # dane opóźnione o 1 dzień
    X2 = M[1 + 2:-31 + 2, :]  # dane opóźnione o 2 dni
    X3 = M[1 + 3:-31 + 3, :]  # dane opóźnione o 3 dni
    X4 = M[1 + 4:-31 + 4, :]  # dane opóźnione o 4 dni
    X5 = M[1 + 5:-31 + 5, :]  # dane opóźnione o 5 dni
    X6 = M[1 + 6:-31 + 6, :]  # dane opóźnione o 6 dni
    X7 = M[1 + 7:-31 + 7, :]  # dane opóźnione o 7 dni czyli sprzed tygodnia
    X10 = M[1 + 10:-31 + 10, :]  # dane opóźnione o 10
    X14 = M[1 + 14:-31 + 14, :]  # dane opóźnione o 14 dni czyli sprzed dwóch tygodni
    X30 = M[1 + 30:-31 + 30, :]  # dane opóźnione o 30 dni czyli sprzed miesiąca
    X = np.hstack((X0, X1, X2, X3, X4, X5, X6, X7, X10, X14, X30))
    # print(X.shape)
    T = M[0:-32, 3]
    # print(T.shape)
    # print(T)
    X_train = X[365:, :]
    T_train = T[365:]
    X_test = X[0:365, :]
    T_test = T[0:365]

    return X_train, T_train, X_test, T_test

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

def use_model2(model_name,Ticker):
    dane = data2(Ticker)
    X_test = dane[2]
    T_test = dane[3]
    model = keras.models.load_model(model_name)
    Y_hat = model.predict(X_test)
    plt.rcParams['figure.figsize'] = [12, 6]  # dla uzyskania wiekszego wykresu
    plt.plot(Y_hat, 'r', label='prognoza kursu')
    plt.plot(T_test, label='rzeczywisty kurs')
    plt.legend()
    plt.grid(True)
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

def canonic_model(X_train, T_train):
    model = Sequential()
    input_shape = (88,)
    model.add(Dense(15, input_shape=input_shape, activation='elu'))
    model.add(Dense(15, activation='elu'))  # 10 neuronow z f.aktywacji elu
    model.add(Dense(10, activation='elu'))  # 10 neuronow z f.aktywacji elu
    model.add(Dense(1, activation='linear'))
    epochs = 50
    model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mean_squared_error'])
    learning = model.fit(X_train, T_train, epochs=epochs,verbose = 1, validation_split = 0.2)
    model.save('canonic_model.h5')
    plt.plot(learning.history['val_loss'], label='f. straty dla zbioruwalidacyjnego')
    plt.xlabel('Epoki')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(model.summary())

    dane = data('AAPL')


# model_regression(dane[0], dane[1],dane[2], dane[3])
#LSTM1(dane[0], dane[2])
#use_model('modelLSTM.h5', 'AAPL')

#dane = data2('AAPL')
# canonic_model(dane[0], dane[1])

use_model2('canonic_model.h5','AAPL')