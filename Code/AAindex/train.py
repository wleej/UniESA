# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train.py
# Time       ：2024/11/14 下午5:57
# Author     ：wleej
# version    ：python 3.10
# Description：
"""
import pickle

from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.metrics import normalized_root_mse as compare_nrmse
from scipy.stats import spearmanr

filepath = "./output_encoding/your_best_encoding.csv"

# load data
data_x = pd.read_csv(filepath, header=None)
data_y = pd.read_csv('your_data.csv', header=None)

# split data
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.20, random_state=15)

model_name = 'your_model_name'
file_model = model_name + '.pickle'

model = GradientBoostingRegressor()
model.fit(x_train, y_train)

with open(file_model, 'wb') as fw:
    pickle.dump(model, fw)

y_pred = pls.predict(x_test)
print(y_test)
print(y_pred)

R2 = round(r2_score(y_test, y_pred),4)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
y_test2 = np.array(y_test.squeeze(axis=None))
y_test3 = y_test2.astype(float)
nrmse = compare_nrmse(y_test3, y_pred)
pccs = pearsonr(y_test3,y_pred)
correlation, p_value = spearmanr(y_test,y_pred)

print('Test RMSE: %.4f , NRMSE: %.4f , R2: %.4f ,R:%s, P:%.4f' % (rmse, nrmse, R2, pccs, correlation))
print('*' * 30)