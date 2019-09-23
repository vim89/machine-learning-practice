#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:36:29 2019

@author: brbonham
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

raw_data = pd.read_csv('/Users/brbonham/Documents/ML Guild/MLGuild_05/2019 County Health Rankings Data - v2.csv')
raw_data_na = raw_data.dropna()

raw_data_names = list(raw_data_na.columns)

del raw_data_names[0:3]

raw_data_na = raw_data_na[raw_data_names]

data_types = raw_data_na.dtypes
pca = PCA(copy=True, iterated_power='auto', n_components=10, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
raw_data_na = raw_data_na[raw_data_names]
pca.fit(raw_data_na)  
print(pca.explained_variance_ratio_)  

sum(pca.explained_variance_ratio_)

pcaed = pca.transform(raw_data_na)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

raw_data_minmax = MinMaxScaler().fit_transform(raw_data_na)



