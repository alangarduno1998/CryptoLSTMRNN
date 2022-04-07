import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
import xgboost as xgb
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.datasets import load_boston
print(xgb.__version__)
df = load_boston()
X = df.data
y = df.target

scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

model = xgb.XGBRegressor(objective='reg:squarederror', verbosity=0)
model.fit(X_train, y_train)

y_pred_val = model.predict(X_test, iteration_range=(0,2))
y_pred_train = model.predict(X_train, iteration_range=(0,2))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f'Shape: {shap_values.shape}')
pd.DataFrame(shap_values).head()

shap.summary_plot(shap_values, X_test,plot_type='bar')

shap.summary_plot(shap_values, X_test)

shap.dependence_plot('close', shap_values, X_test)

shap.dependence_plot('high', shap_values, X_test)

shap.initjs()

shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

print('Feature 12')
shap.force_plot(explainer.expected_value, shap_values[12], X_test.iloc[12])
print('Feature 20')
shap.force_plot(explainer.expected_value, shap_values[20], X_test.iloc[20])
print('Feature 47')
shap.force_plot(explainer.expected_value, shap_values[47], X_test.iloc[47])