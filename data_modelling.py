############### MODELLING ###############

import pandas as pd
import numpy as np
vehicles = pd.read_csv('Data/train_vehicles_models.csv') # Uploading the data
vehicles.dtypes




# LGBOOST MODEL LEARNING # 
import lightgbm as lgb

x_train = vehicles.drop('price', axis=1)
y_train = vehicles['price']

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

train_data = lgb.Dataset(x_train, label=y_train)

num_round = 100
lgbm_model = lgb.train(params, train_data, num_round, valid_sets=[val_data], early_stopping_rounds=10)