#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 22:59:45 2019

@author: brbonham
"""

from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import os
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
import statsmodels.api as sm
from statsmodels.api import OLS
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
path = '/Users/brbonham/Documents/ML Guild/MLGuild_05/'
os.chdir(path)
os.getcwd()

# Great Duke artical on fitting an ARIMA
#https://people.duke.edu/~rnau/411arim3.htm

#Setting to datetime (if you want to use)
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = pd.read_csv('05_shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

#Plotting the time_series
series.plot()
pyplot.show()

#Initial ACF Test
autocorrelation_plot(series)
pyplot.show()

#Fitting a 0, 1, 0 ARIMA
model = ARIMA(series, order=(0,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

#Plotting Residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

#Splitting the Data
X = series.values
train_series, test_series = series[0:-12], series[-12:]
train, test = X[0:-12], X[-12:]

#Dickey Fuller Test
result = adfuller(X)

#Exponential Smoothing Model
model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit()
pred = fit.forecast(12)
sse1 = np.sqrt(np.mean(np.square(test - pred)))

#Seasonal ARIMA Code
sarima_model = SARIMAX(train, order=(0, 1, 2), seasonal_order=(0, 1, 2, 12), enforce_invertibility=False, enforce_stationarity=False)
sarima_fit = sarima_model.fit()

fig, ax = plt.subplots(1, 1, sharey=True, figsize=(12, 6))
autocorrelation_plot(model_fit.resid)

#PACF and ACF Plots
fig, ax = plt.subplots(2, figsize=(12,6))
ax[0] = plot_acf(model_fit.resid, ax=ax[0], lags=20)
ax[1] = plot_pacf(model_fit.resid, ax=ax[1], lags=20)


