import sys
import pandas as pd
import math
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, LSTM, ConvLSTM1D, Dropout, GRU, Bidirectional
from sklearn.preprocessing import scale, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import warnings
warnings.filterwarnings("ignore")

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return rmse

def Scaler(*x):
    for idx, data in enumerate(x):
        if idx == 0:
            xm_slr = RobustScaler().fit(data)
            xm_data = xm_slr.transform(data)
            if len(x) == 1:
                return xm_slr, xm_data
        elif idx == 1:
            xa_slr = RobustScaler().fit(data)
            xa_data = xa_slr.transform(data)
            return xm_slr, xm_data, xa_slr, xa_data

def split_sequence(x_data, y_data, prior, after):
    raw_data = np.hstack((x_data, y_data))

    x, y = [], []
    for i in range(len(raw_data)):
        in_end_idx = i + prior
        out_end_idx = in_end_idx + after

        if out_end_idx > len(raw_data):
            break

        seq_x, seq_y = raw_data[i:in_end_idx, 1], raw_data[in_end_idx:out_end_idx, -1]
        x.append(seq_x)
        y.append(seq_y)

    x = np.array(x)
    y = np.array(y)

    x = x.reshape(x.shape[0], x.shape[1],1)
    return x, y

def plot_lstm_response(interval_tr, interval_te, x_train, y_train, x_test, y_test, pred, prior, name):
    plt.figure(1)
    plt.plot(interval_tr, y_train, ':',color='black',label='Experiment-training')
    plt.plot(interval_te, t_y_test_raw[prior:], ':',color='red',label='Experiment-test')
    plt.plot(interval_te, pred, ':',color='blue',label='Predicted')
    plt.title('R2: {} \n RMSE: {}'.format(round(r2_score(y_test[prior:],pred),4) ,round(return_rmse(y_test[prior:],pred),4)))
    plt.xlabel('Time [sec]', fontsize=15)
    plt.ylabel('Displacement [mm]', fontsize=15)
    plt.grid()
    plt.legend()
    plt.savefig(name+'_time-force.png')
    plt.close()
    return

n_lookback = 'Put HT results'
n_forecast = 'Put HT results'
n_1st = 'Put HT results'
n_2nd = 'Put HT results'
n_fnn = 'Put HT results'
GMdata = 'CHAN001'
DRdata = 'CHAN001'

GM_1 = pd.read_excel('source path/' + GMdata + '.xlsx', header=0, usecols="A,B")
DR_1 = pd.read_excel('source path/' + DRdata + '.xlsx', header=0, usecols="C")

data1 = pd.concat([GM_1, DR_1], axis=1)

num_test = int(GM_1.shape[0]*0.5)

time_train, time_test = data1[:-num_test], data1[-num_test:]
time_test = pd.concat([time_train[-n_lookback:], time_test])

t_x_train_raw = time_train['acc'].values.reshape(-1,1)
t_y_train_raw = time_train['disp'].values.reshape(-1,1)
t_x_slr, t_x_train, t_y_slr, t_y_train = Scaler(t_x_train_raw, t_y_train_raw)

t_x_test_raw = time_test['acc'].values.reshape(-1,1)
t_y_test_raw = time_test['disp'].values.reshape(-1,1)
t_x_test = t_x_slr.transform(t_x_test_raw)
t_y_test = t_y_slr.transform(t_y_test_raw)

x_train, y_train = split_sequence(t_x_train, t_y_train, n_lookback, n_forecast)
x_test, y_test = split_sequence(t_x_test, t_y_test, n_lookback, n_forecast)

pre_lstm_input = Input(shape=(n_lookback,x_train.shape[2]), name='main_input')
pre_lstm_l1 = LSTM(units=n_1st, return_sequences=True)(pre_lstm_input)
pre_lstm_l2 = LSTM(units=n_2nd, return_sequences=False)(pre_lstm_l1)
pre_lstm_fnn = Dense(n_fnn)(pre_lstm_l2)
pre_lstm_output = Dense(n_forecast, name='output')(pre_lstm_fnn)
pre_lstm_model = Model(inputs=pre_lstm_input, outputs=pre_lstm_output)

pre_lstm_model.summary()
pre_lstm_model.compile(optimizer='adam',loss='mean_squared_error')
pre_lstm_model.fit(x_train,y_train,epochs=10,batch_size=50)

lstm_pred = pre_lstm_model.predict(x_test).reshape(-1,1)
lstm_pred = t_y_slr.inverse_transform(lstm_pred)

plot_lstm_response(time_train['time'], time_test['time'].iloc[n_lookback:], t_x_train_raw, t_y_train_raw, t_x_test_raw, t_y_test_raw, lstm_pred, n_lookback, 'EQ1')

pre_test = pd.DataFrame({
    'time': time_test['time'].iloc[n_lookback:],
    'acc': time_test['acc'].iloc[n_lookback:],
    'disp': time_test['disp'].iloc[n_lookback:],
    'pred': lstm_pred.flatten()
})

GM_2 = pd.read_excel('target path/' + GMdata + '.xlsx', header=0, usecols="A,B")
DR_2 = pd.read_excel('target path/' + DRdata + '.xlsx', header=0, usecols="C")

data2 = pd.concat([GM_2, DR_2], axis=1)

num_test = int(data2.shape[0]*0.95)
time_train, time_test = data2[:-num_test], data2[-num_test:]
time_test = pd.concat([time_train[-n_lookback:], time_test])

t_x_train_raw = time_train['acc'].values.reshape(-1,1)
t_y_train_raw = time_train['disp'].values.reshape(-1,1)
t_x_train = t_x_slr.transform(t_x_train_raw)
t_y_train = t_y_slr.transform(t_y_train_raw)

t_x_test_raw = time_test['acc'].values.reshape(-1,1)
t_y_test_raw = time_test['disp'].values.reshape(-1,1)
t_x_test = t_x_slr.transform(t_x_test_raw)
t_y_test = t_y_slr.transform(t_y_test_raw)

x_train, y_train = split_sequence(t_x_train, t_y_train, n_lookback, n_forecast)
x_test, y_test = split_sequence(t_x_test, t_y_test, n_lookback, n_forecast)

new_lstm_input = Input(shape=(n_lookback,x_train.shape[2]), name='main_input')
new_lstm_l1 = LSTM(units=n_1st, return_sequences=True)(new_lstm_input)
new_lstm_l2 = LSTM(units=n_2nd, return_sequences=False)(new_lstm_l1)
new_lstm_fnn = Dense(n_fnn)(new_lstm_l2)
new_lstm_output = Dense(n_forecast, name='output')(new_lstm_fnn)
new_lstm_model = Model(inputs=new_lstm_input, outputs=new_lstm_output)

new_lstm_model.layers[1].set_weights(pre_lstm_model.layers[1].get_weights())
new_lstm_model.layers[1].trainable=False
new_lstm_model.layers[2].set_weights(pre_lstm_model.layers[2].get_weights())
new_lstm_model.layers[2].trainable=False

new_lstm_model.compile(optimizer='adam',loss='mean_squared_error')
new_lstm_model.summary()

new_lstm_model.fit(x_train,y_train,epochs=10,batch_size=5)

lstm_pred2 = new_lstm_model.predict(x_test).reshape(-1,1)
lstm_pred2 = t_y_slr.inverse_transform(lstm_pred2)

plot_lstm_response(time_train['time'], time_test['time'].iloc[n_lookback:], t_x_train_raw, t_y_train_raw, t_x_test_raw, t_y_test_raw, lstm_pred2, n_lookback, 'EQ2')

pre_test = pd.DataFrame({
    'time': time_test['time'].iloc[n_lookback:],
    'acc': time_test['acc'].iloc[n_lookback:],
    'disp': time_test['disp'].iloc[n_lookback:],
    'pred': lstm_pred2.flatten()
})