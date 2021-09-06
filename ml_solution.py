import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time, datetime as dt
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import sys

import alpha
import alpaca
import database as db
from populate import download_data
from portfolios import Portfolio
from history import *

DataStore = db.DataStore()

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression

dates = [dt.datetime(2008,1,1), dt.datetime(2015,12,31)]
dates = None
sym = 'JPM'
indicators = ['SMA','OBV', 'AD', 'BBANDS', 'MFI', 'SAR', 'T3', 'MOM', 'MIDPRICE', 'WMA']
indicators = 'all'
history = History(DataStore, sym, dates, indicators=indicators)
indicators = history.indicators
prices = history.prices

indicators.rename(columns={'adjusted close': 'Close'}, inplace=True)

predictors = indicators.iloc[:, 1:]
close_price = indicators.iloc[:, 0]


# predict tomorrow's closing price using today's indicators... 
predictors = predictors[:-1]
close_price = close_price[1:]

scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)

X_train, X_test, y_train, y_test = train_test_split(predictors, close_price, test_size=0.3)

y_train = y_train.to_numpy().astype(float)
y_test = y_test.to_numpy().astype(float)


# clf = tree.DecisionTreeRegressor()
# clf = clf.fit(X_train, y_train)
# clf.score(X_test, y_test)

# from sklearn import svm
# regr = svm.SVR()
# regr = regr.fit(X_train, y_train)
# regr.score(X_test, y_test)

X, y = make_regression(n_features=4, n_informative=2,shuffle=False)
regr = AdaBoostRegressor(n_estimators=100)
regr.fit(X, y)
print(regr.score(X, y))