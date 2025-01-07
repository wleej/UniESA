# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : compare.py
# Time       ：2024/10/19 下午6:18
# Author     ：wleej
# version    ：python 3.10
# Description：
"""
import pickle
from sklearn.linear_model import ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score, mean_squared_error

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.metrics import normalized_root_mse as compare_nrmse
from scipy.stats import spearmanr

filter = [".csv"]
FitnessBest = []
x_type = []

def all_path(dirname):
  for maindir, subdir, file_name_list in os.walk(dirname):
    for filename in file_name_list:
      x_type.append(filename)
      apath = os.path.join(maindir, filename)
      FitnessBest.append(apath)
  return FitnessBest

print(all_path(r"output_encoding_path"))

R2_list = []
p=0

for i in FitnessBest:
  filepath = i

  # load data
  data_x = pd.read_csv(filepath,header = None)
  data_y = pd.read_csv('your_data.csv',  header=None)

  # split data
  x_train, x_test, y_train, y_test = train_test_split( data_x, data_y, test_size=0.20, random_state=15)

  # model fit ： You can choose different algorithms for the model.
  model = GradientBoostingRegressor()
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)

  # calculate R2, rmse, mae, pccs
  R2 = round(r2_score(y_test, y_pred),4)
  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(y_test, y_pred)
  y_test2 = np.array(y_test.squeeze(axis=None))
  y_test3 = y_test2.astype(float)
  nrmse = compare_nrmse(y_test3, y_pred)
  pccs = pearsonr(y_test3,y_pred)
  correlation, p_value = spearmanr(y_test,y_pred)
  R2_list.append(R2)

  print(p)
  print('Test RMSE: %.4f , NRMSE: %.4f , R2: %.4f ,R:%s, P:%.4f' % (rmse, nrmse, R2, pccs, correlation))
  p=p+1

print('*' * 30)
print('R2:',max(R2_list))
print('encoding_now',x_type[R2_list.index(max(R2_list))])