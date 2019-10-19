from keras.layers import Dropout, Activation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def add_original_feature(df, df_new):
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)
    df_new['crudeoil_1'] = df['crudeoil'].shift(1)
    df_new['gold_1'] = df['gold'].shift(1)


def add_avg_price(df, df_new):
    df_new['avg_price_5'] = df['Close'].rolling(5).mean().shift(1)
    df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)
    df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new[
        'avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new[
        'avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new[
        'avg_price_365']


def add_avg_volume(df, df_new):
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new[
        'avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new[
        'avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new[
        'avg_volume_365']


def add_avg_crudeoil(df, df_new):
    df_new['avg_crudeoil_5'] = df['crudeoil'].rolling(5).mean().shift(1)
    df_new['avg_crudeoil_30'] = df['crudeoil'].rolling(21).mean().shift(1)
    df_new['avg_crudeoil_365'] = df['crudeoil'].rolling(252).mean().shift(1)
    df_new['ratio_avg_crudeoil_5_30'] = df_new['avg_crudeoil_5'] / df_new[
        'avg_crudeoil_30']
    df_new['ratio_avg_crudeoil_5_365'] = df_new['avg_crudeoil_5'] / df_new[
        'avg_crudeoil_365']
    df_new['ratio_avg_crudeoil_30_365'] = df_new['avg_crudeoil_30'] / df_new[
        'avg_crudeoil_365']


def add_avg_gold(df, df_new):
    df_new['avg_gold_5'] = df['gold'].rolling(5).mean().shift(1)
    df_new['avg_gold_30'] = df['gold'].rolling(21).mean().shift(1)
    df_new['avg_gold_365'] = df['gold'].rolling(252).mean().shift(1)
    df_new['ratio_avg_gold_5_30'] = df_new['avg_gold_5'] / df_new['avg_gold_30']
    df_new['ratio_avg_gold_5_365'] = df_new['avg_gold_5'] / df_new[
        'avg_gold_365']
    df_new['ratio_avg_gold_30_365'] = df_new['avg_gold_30'] / df_new[
        'avg_gold_365']


def add_std_price(df, df_new):
    df_new['std_price_5'] = df['Close'].rolling(5).std().shift(1)
    df_new['std_price_30'] = df['Close'].rolling(21).std().shift(1)
    df_new['std_price_365'] = df['Close'].rolling(252).std().shift(1)
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new[
        'std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new[
        'std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new[
        'std_price_365']


def add_std_volume(df, df_new):
    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)
    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new[
        'std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new[
        'std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new[
        'std_volume_365']


def add_std_crudeoil(df, df_new):
    df_new['std_crudeoil_5'] = df['crudeoil'].rolling(5).std().shift(1)
    df_new['std_crudeoil_30'] = df['crudeoil'].rolling(21).std().shift(1)
    df_new['std_crudeoil_365'] = df['crudeoil'].rolling(252).std().shift(1)
    df_new['ratio_std_crudeoil_5_30'] = df_new['std_crudeoil_5'] / df_new[
        'std_crudeoil_30']
    df_new['ratio_std_crudeoil_5_365'] = df_new['std_crudeoil_5'] / df_new[
        'std_crudeoil_365']
    df_new['ratio_std_crudeoil_30_365'] = df_new['std_crudeoil_30'] / df_new[
        'std_crudeoil_365']


def add_std_gold(df, df_new):
    df_new['std_gold_5'] = df['gold'].rolling(5).std().shift(1)
    df_new['std_gold_30'] = df['gold'].rolling(21).std().shift(1)
    df_new['std_gold_365'] = df['gold'].rolling(252).std().shift(1)
    df_new['ratio_std_gold_5_30'] = df_new['std_gold_5'] / df_new['std_gold_30']
    df_new['ratio_std_gold_5_365'] = df_new['std_gold_5'] / df_new[
        'std_gold_365']
    df_new['ratio_std_gold_30_365'] = df_new['std_gold_30'] / df_new[
        'std_gold_365']


def add_return_feature(df, df_new):
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) /
                          df['Close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['Close'] - df['Close'].shift(5)) /
                          df['Close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) /
                           df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) /
                            df['Close'].shift(252)).shift(1)
    df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean().shift(1)
    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean().shift(1)
    df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean().shift(1)


def generate_features(df):
    df_new = pd.DataFrame()
    add_original_feature(df, df_new)
    add_avg_price(df, df_new)
    add_avg_volume(df, df_new)
    add_avg_crudeoil(df, df_new)
    add_avg_gold(df, df_new)
    add_return_feature(df, df_new)
    df_new['close'] = df['Close']
    df_new = df_new.dropna(axis=0)
    return df_new


a = pd.read_csv('/home/sklakshminarayanan/stock_data/crudeoil.csv',
                index_col='Date')
b = pd.read_csv('/home/sklakshminarayanan/stock_data/gold.csv',
                index_col='Date')
a = pd.merge(a, b, on='Date', how='inner')
data_raw = pd.read_csv('/home/sklakshminarayanan/stock_data/stock.csv',
                       index_col='Date')
data_raw = pd.merge(data_raw, a, on='Date', how='inner')
data = generate_features(data_raw)
print(data.shape)

values = data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

n_train_time = 750
train = scaled[:n_train_time, :]
test = scaled[n_train_time:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(512, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer="Adam")

# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=60,
                    validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LSTM Advanced Model with Moving Average Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('/home/sklakshminarayanan/stock_data/model/lstm_mv_ma_loss.png')
plt.show()

yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], 39))
temp = test_X[:, -39:]
inv_yhat = np.concatenate((yhat, temp), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -39:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
mse = mean_squared_error(inv_y, inv_yhat)
mae = mean_absolute_error(inv_y, inv_yhat)
r2 = r2_score(inv_y, inv_yhat)
print('Test RMSE: %.3f' % rmse)
print('Test MSE: %.3f' % mse)
print('Test MAE: %.3f' % mae)
print('Test R2 score: %.3f' % r2)

loss = {}
loss['rmse'] = rmse
loss['mse'] = mse
loss['mae'] = mae
loss['r2'] = r2
loss = pd.DataFrame(loss, index=[0])
loss.to_csv('/home/sklakshminarayanan/stock_data/data/lstm_mv_ma_loss.csv',
            index=None)

lstm1 = {}
lstm1['pred'] = inv_yhat
lstm1['test'] = inv_y
lstm_mv = pd.DataFrame(lstm1)
lstm_mv.to_csv('/home/sklakshminarayanan/stock_data/data/lstm_mv_ma_pred.csv',
               index=None)

plt.plot(inv_y)
plt.plot(inv_yhat)
plt.title('LSTM Advanced Model with Moving Average Test vs Prediction')
plt.ylabel('close')
plt.xlabel('date')
plt.legend(['test', 'predicted'], loc='lower left')
plt.savefig(
    '/home/sklakshminarayanan/stock_data/model/lstm_mv_ma_testvspred.png')
plt.show()

b = pd.DataFrame(yhat)
b['test'] = b[0]
c = pd.DataFrame(test_y)
b['Predicted'] = c[0]
a = pd.DataFrame(train_y)
a = a.append(b, ignore_index=True)
plt.plot(a[0])
plt.plot(a['Predicted'])
plt.plot(a['test'])
plt.title('LSTM Advanced Model with Moving Average Model Prediction')
plt.ylabel('stock price')
plt.xlabel('date')
plt.legend(['train', 'test', 'predicted'], loc='upper left')
plt.savefig('/home/sklakshminarayanan/stock_data/model/lstm_mv_ma_predfull.png')
plt.show()
