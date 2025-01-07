# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train2.py
# Time       ：2024/9/24 下午3:06
# Author     ：wleej
# version    ：python 3.10
# Description：
"""
import pickle
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error,r2_score

import lightgbm as lgb
import xgboost as xgb                        #xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
from sklearn.linear_model import ElasticNet  #ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False)
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR    # from sklearn.svm import SVC   model = SVC(kernel='rbf', probability=True)
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

data = []
for r in range(140):
    c = r + 1
    file = './data/' + str(c) + '.pt'
    loaded_dict = {}
    loaded_dict = torch.load(file)
    a = loaded_dict['mean_representations']
    data.append(a[33])

final_tensor = torch.stack(data).detach()
proteins = torch.flatten(final_tensor,1)

model_name = 'your_model_name'
file_model = './model/output/' + model_name + '.pickle'
dir_input = './data/'

your_fitness = load_pickle(dir_input + 'fitness.pickle')
dep = torch.FloatTensor(your_fitness)

random = 1
epoch = 100

for random in range(epoch):
    x_train, x_test, y_train, y_test = train_test_split(proteins, dep, test_size=0.2, random_state= random)

# 创建并训练模型
    model = BayesianRidge()
    deppredivt= model.fit(x_train, y_train)

    with open(file_model, 'wb') as fw:
        pickle.dump(deppredivt, fw)

# 调用.pickle
    with open(file_model, 'rb') as fr:
        new = pickle.load(fr)
        y_pred = deppredivt.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        pccs = pearsonr(y_test, y_pred)
        print('random:%d' %random)
        print('Test RMSE: %.4f , MAE: %.4f , R2: %.4f ,PCC:%s' % (rmse, mae, r2, pccs))
