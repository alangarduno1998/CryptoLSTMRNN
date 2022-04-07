import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
sns.set_style("white")
import scipy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, Activation
from tensorflow.python.keras.layers import CuDNNLSTM, CuDNNGRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from IPython.core.pylabtools import figsize
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Set random seed for reproducibility
tf.random.set_seed(42)

pair = "BTC-USD"
pair_split = pair.split('-')  # must split in format XXX/XXX
#csv_path =r'C:\Users\alang\PycharmProjects\CryptoLSTMRNN\Coinbasedata\{}-{} (1).csv'.format(pair_split[0], pair_split[1])
csv_path =r'C:\Users\alang\PycharmProjects\CryptoLSTMRNN\Coinbasedata\Bitcoin_price.csv'
df = pd.read_csv(csv_path,parse_dates=['Date'])
df['Date'] = df['Date'].astype('datetime64[ns]')
print(df.head())
print(df.shape)

# Create variable 'TOMORROW_CLOSE' which shifts 'Close' up by 1
df['TOMORROW_CLOSE'] = df['Close'].shift(-1,fill_value=0)
df.drop(df.tail(1).index,inplace=True)
df = df.drop(columns=['Close'])
df = df.set_index('Date')

# Split train data (80%) and test data (20%)
train_size = int(len(df)*0.8)
train_dataset, test_dataset = df.iloc[:train_size],df.iloc[train_size:]

# Plot train and test data
def plot_data(train_dataset, test_dataset, pair):
    plt.figure(figsize = (12, 6))
    plt.rcParams['figure.dpi'] = 360
    plt.plot(train_dataset.TOMORROW_CLOSE)
    plt.plot(test_dataset.TOMORROW_CLOSE)
    plt.title('Price data for ' + pair)
    plt.xlabel('Date')
    plt.ylabel('Close value (US$)')
    plt.legend(['Train set', 'Test set'], loc='upper left')


plot_data(train_dataset, test_dataset, pair)
print('Dimension of train data: ',train_dataset.shape)
print('Dimension of test data: ', test_dataset.shape)

# Split train data to X and y
X_train = train_dataset.drop('TOMORROW_CLOSE', axis = 1)
y_train = train_dataset.loc[:,['TOMORROW_CLOSE']]
# Split test data to X and y
X_test = test_dataset.drop('TOMORROW_CLOSE', axis = 1)
y_test = test_dataset.loc[:,['TOMORROW_CLOSE']]

# Different scaler for input and output
scaler_x = MinMaxScaler(feature_range = (-1,1))
scaler_y = MinMaxScaler(feature_range = (-1,1))
# Fit the scaler using available training data
input_scaler = scaler_x.fit(X_train)
output_scaler = scaler_y.fit(y_train)
# Apply the scaler to training data
train_y_norm = output_scaler.transform(y_train)
train_x_norm = input_scaler.transform(X_train)
# Apply the scaler to test data
test_y_norm = output_scaler.transform(y_test)
test_x_norm = input_scaler.transform(X_test)

def threeD_dataset(X, y, time_steps=1):
    Xs, ys = [], []

    for i in range(len(X) - time_steps):
        v = X[i:i + time_steps, :]
        Xs.append(v)
        ys.append(y[i + time_steps])

    return np.array(Xs), np.array(ys)


TIME_STEPS = 10
X_test, y_test = threeD_dataset(test_x_norm, test_y_norm, TIME_STEPS)
X_train, y_train = threeD_dataset(train_x_norm, train_y_norm, TIME_STEPS)
print('X_test.shape: ', X_test.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape)
print('y_train.shape: ', y_train.shape)

# Create BiLSTM model
def create_model_bilstm(units):
    model = Sequential()
    # First layer of BiLSTM
    model.add(Bidirectional(LSTM(units = units, return_sequences=True),
                            input_shape=(X_train.shape[1], X_train.shape[2])))
    # Second layer of BiLSTM
    model.add(Bidirectional(LSTM(units = units)))
    model.add(Dense(1))
    #Compile model
    model.compile(loss='mse', optimizer='adam')
    return model


# Create LSTM or GRU model
def create_model(units, m):
    model = Sequential()
    # First layer of LSTM
    model.add(m (units = units, return_sequences = True,
                 input_shape = [X_train.shape[1], X_train.shape[2]]))
    model.add(Dropout(0.2))
    # Second layer of LSTM
    model.add(m (units = units))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    #Compile model
    model.compile(loss='mse', optimizer='adam')
    return model


# BiLSTM
model_bilstm = create_model_bilstm(64)

# GRU and LSTM
model_gru = create_model(64, GRU)
model_lstm = create_model(64, LSTM)

# Fit BiLSTM, LSTM and GRU
def fit_model(model):
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 10)

    # shuffle = False because the order of the data matters
    history = model.fit(X_train, y_train, epochs = 100, validation_split = 0.2,
                    batch_size = 32, shuffle = False, callbacks = [early_stop], verbose=2)
    return history


history_bilstm = fit_model(model_bilstm)
history_lstm = fit_model(model_lstm)
history_gru = fit_model(model_gru)
# Plot train and validation loss and accuracy
# def plot_loss(history, model_name):
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model Train vs Validation Loss for ' + model_name)
#     plt.ylabel('Loss')
#     plt.xlabel('epoch')
#     plt.legend(['Train loss', 'Validation loss'], loc='upper right')
#
#
# plot_loss(history_bilstm, 'BiLSTM')
# plot_loss(history_lstm, 'LSTM')
# plot_loss(history_gru, 'GRU')
# plt.show()

def plot_acc(history, model_name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Train vs Validation Accuracy for ' + model_name)
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train accuracy', 'Validation accuracy'], loc='upper right')

plot_acc(history_bilstm, 'BiLSTM')
plot_acc(history_lstm, 'LSTM')
plot_acc(history_gru, 'GRU')
plt.show();

# Note that I have to use scaler_y
y_test = scaler_y.inverse_transform(y_test)
y_train = scaler_y.inverse_transform(y_train)
def prediction(model):
    prediction = model.predict(X_test)
    prediction = scaler_y.inverse_transform(prediction)
    return prediction


prediction_bilstm = prediction(model_bilstm)
prediction_lstm = prediction(model_lstm)
prediction_gru = prediction(model_gru)

def plot_future(prediction, model_name, y_test):
    range_future = len(prediction)
    plt.plot(np.arange(range_future), np.array(y_test), label='True Future')
    plt.plot(np.arange(range_future), np.array(prediction), label='Prediction')
    plt.title('True future vs prediction for ' + model_name)
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Close value (USD $)')


plot_future(prediction_bilstm, 'BiLSTM', y_test)
plot_future(prediction_lstm, 'LSTM', y_test)
plot_future(prediction_gru, 'GRU', y_test)
plt.show()

# Define a function to calculate MAE and RMSE
def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()

    print(model_name + ':')
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    print('')

evaluate_prediction(prediction_bilstm, y_test, 'Bidirectional LSTM')
evaluate_prediction(prediction_lstm, y_test, 'LSTM')
evaluate_prediction(prediction_gru, y_test, 'GRU')