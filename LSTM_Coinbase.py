import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
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
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, BatchNormalization
from tensorflow.python.keras.layers import CuDNNLSTM
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
# csv_path=r'C:\Users\alang\PycharmProjects\CryptoLSTMRNN\Coinbasedata\ETH-USD (1).csv'
csv_path=r'C:\Users\alang\PycharmProjects\CryptoLSTMRNN\Coinbasedata\Ethereum_price.csv'
df = pd.read_csv(csv_path,parse_dates=['Date'])
df = df.sort_values('Date')
df = df[['Date','Open','High','Low','Volume','MarketCap','Close']]
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.figure(figsize=(15, 5))
plt.title('ETH/USD price data')
plt.plot(df.Date, df.Close)
plt.xlabel("Date")
# plt.ylabel('Close Price ({})'.format(pair_split[1]))
# plt.legend()

#Normalization
scaler = MinMaxScaler()
close_price = df.Close.values.reshape(-1, 1) #scaler expects data in form x,y so dummy dimension is added
scaled_close = scaler.fit_transform(close_price)
print(scaled_close.shape,'\n',np.isnan(scaled_close).any())
scaled_close = scaled_close[~np.isnan(scaled_close)]
scaled_close = scaled_close.reshape(-1, 1) # isnan to filter out NaN values
print(np.isnan(scaled_close).any(),'\n')

#preprocessing
SEQ_LEN = 60

def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

def preprocess(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])
    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = preprocess(scaled_close, SEQ_LEN, train_split = 0.8)
#we save 20% of data for testing using 2212 sequences representing 253 days of ETHEREUM price and predicting 90 days in future

print(X_train.shape)
print(X_test.shape)
DROPOUT = 0.2 # use dropout with rate of 20% to combat overfitting during training
WINDOW_SIZE = SEQ_LEN - 1
print(WINDOW_SIZE, X_train.shape[-1])

model = Sequential()
model.add(Bidirectional(CuDNNLSTM(WINDOW_SIZE, return_sequences=True),
                        input_shape=(WINDOW_SIZE,X_train.shape[-1])))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(CuDNNLSTM((WINDOW_SIZE * 2), return_sequences=True))) # allows you to train on sequence data in forward and reversed direction
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(CuDNNLSTM(WINDOW_SIZE, return_sequences=False)))
model.add(Dropout(rate=DROPOUT))
model.add(BatchNormalization())
model.add(Dense(units=1))

model.add(Activation('linear')) #linear activation function

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=["accuracy"],
)
BATCH_SIZE = 64
print(model.summary())

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=2,
    validation_split=0.4
)
train_acc = model.evaluate(X_train, y_train, verbose=0)
test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc[1], test_acc[1]))
model.evaluate(X_test, y_test)


plt.subplot(211)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.subplot(212)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show();

#Prediction
y_hat = model.predict(X_test)

y_test_inverse = scaler.inverse_transform(y_test)
y_hat_inverse = scaler.inverse_transform(y_hat)

plt.plot(y_test_inverse, label="Actual Price", color='green')
plt.plot(y_hat_inverse, label="Predicted Price", color='red')

plt.title('Ethereum price prediction')
plt.xlabel('Time [days]')
plt.ylabel('Price')
plt.legend(loc='best')


def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()

    print(model_name + ':')
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    print('')

evaluate_prediction(y_hat_inverse, y_test_inverse, 'Bidirectional LSTM')
plt.show();