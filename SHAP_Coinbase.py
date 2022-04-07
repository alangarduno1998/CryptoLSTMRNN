import os
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

df = pd.read_csv(r'C:\Users\alang\PycharmProjects\CryptoLSTMRNN\Coinbasedata\Ethereum_price.csv',parse_dates=['Date'])
df = df.sort_values('Date')
df['Date'] = pd.to_numeric(df['Date'])
df = df[['Date','Open','High','Low','Volume','MarketCap','Close']]
# df = df[['low','high','open','close','volume','vol_fiat']]
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

model = xgb.XGBRegressor(max_depth=3, n_estimators=300, colsample_bytree=0.75, seed=123)
model.fit(X_train, y_train)

y_pred_val = model.predict(X_test)
y_pred_train = model.predict(X_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f'Shape: {shap_values.shape}')
pd.DataFrame(shap_values).head()
shap.initjs()
shap.summary_plot(shap_values, X_test,plot_type='bar')

shap.summary_plot(shap_values, X_test)

shap.dependence_plot("High", shap_values, X_test)

shap.dependence_plot("MarketCap", shap_values, X_test.values, feature_names=X.columns)

shap.dependence_plot("rank(1)", shap_values, X_test)

shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True)

print('Feature 12')
shap.force_plot(explainer.expected_value, shap_values[12,:], X_test.iloc[12,:], matplotlib=True)
print('Feature 20')
shap.force_plot(explainer.expected_value, shap_values[20,:], X_test.iloc[20,:], matplotlib=True)
print('Feature 47')
shap.force_plot(explainer.expected_value, shap_values[47,:], X_test.iloc[47,:], matplotlib=True)
help(shap.force_plot)