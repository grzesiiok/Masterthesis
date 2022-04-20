import yfinance as yf
from stocksymbol import StockSymbol
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def namelist():
    try:
        api_key = 'a94d0603-da1a-40a1-830d-19619a07bfc6'
        ss = StockSymbol(api_key)
        list = ss.get_symbol_list(market="US",symbols_only=True)
        return list
    except:
        print("błąd wczytywania listy")

def allstock_print():
    names = namelist()
    try:
        for name in names:
            # hist = yf.Ticker(name).history(period=max)
            # print(hist)
            if "." in name:
                continue
            print(name)
            print(pd.DataFrame(yf.Ticker(name).history(period='max')))
    except:
        print("błąd printowania danych")

def allstockvector():
    names = namelist()
    tab = []
    try:
        for name in names:
            if "." in name:
                continue
            tab.append(pd.DataFrame(yf.Ticker(name).history(period='max')))
            # print(pd.DataFrame(yf.Ticker(name).history(period='max')))
        return tab
    except:
        print("błąd pobierania danych giełdowych")

class firm_data:
    def __init__(self,name):
        self.name = name

    def firm(self):
        try:
            return pd.DataFrame(yf.Ticker(self.name).history(period='max'))
        except:
            print("błąd pobrania danych firmy")


    def heatmap(self):
        try:
            a = firm_data(self.name).firm()
            cormap = a.corr()
            fig, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(cormap, annot=True)
            return plt.show()
        except:
            print("błąd heatmapy")

    def get_corelated_col(self, threshold):
        a = firm_data(self.name).firm()
        cor_dat = a.corr()['Close']
        # Cor_data to be column along which corelation to be measured
        #Threshold be the value above which of corelation to considered
        feature=[]
        value=[]

        for i ,index in enumerate(cor_dat.index):
            if abs(cor_dat[index]) > threshold:
                feature.append(index)
                value.append(cor_dat[index])

        df = pd.DataFrame(data = value, index = feature, columns=['corr value'])
        return df

    def top_corelated_values(self, threshold):
        top_corelated_value = firm_data(self.name).get_corelated_col(threshold)
        df = firm_data(self.name).firm()
        df = df[top_corelated_value.index]
        return df.shape, df.head()

    def plot(self, threshold):
        top_corelated_values = firm_data(self.name).get_corelated_col(threshold)
        df = firm_data(self.name).firm()
        df = df[top_corelated_values.index]
        sns.pairplot(df)
        plt.tight_layout()
        plt.show()


    def minmax(self):
        X = firm_data(self.name).firm().drop(['Close'], axis=1)
        y = firm_data(self.name).firm()['Close']
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        return X, y, X.head()

#print(firm_data('AAPL').minmax()[0])