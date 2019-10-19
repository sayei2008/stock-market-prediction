import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR


def add_original_feature(df, df_new):
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)


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
    add_return_feature(df, df_new)
    df_new['close'] = df['Close']
    df_new = df_new.dropna(axis=0)
    return df_new


data_raw = pd.read_csv('/home/sklakshminarayanan/stock_data/stock.csv',
                       index_col='Date')
data = generate_features(data_raw)
start_train = '2015-01-01'
end_train = '2017-12-21'
start_test = '2017-12-22'
end_test = '2018-12-31'
data_train = data.ix[start_train:end_train]
X_train = data_train.drop('close', axis=1).values
y_train = data_train['close'].values
data_test = data.ix[start_test:end_test]
X_test = data_test.drop('close', axis=1).values
y_test = data_test['close'].values
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

scaler = StandardScaler()

X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)
svr = SVR(kernel='linear')
svr.fit(X_scaled_train, y_train)
yhat = svr.predict(X_scaled_test)

rmse = np.sqrt(mean_squared_error(y_test, yhat))
mse = mean_squared_error(y_test, yhat)
mae = mean_absolute_error(y_test, yhat)
r2 = r2_score(y_test, yhat)
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
loss.to_csv('/home/sklakshminarayanan/stock_data/data/svr_uv_ma_loss.csv',
            index=None)

svr1 = {}
svr1['pred'] = yhat
svr1['test'] = y_test
svr_uv = pd.DataFrame(svr1)
svr_uv.to_csv('/home/sklakshminarayanan/stock_data/data/svr_uv_ma_pred.csv',
              index=None)

plt.plot(y_test)
plt.plot(yhat)
plt.title('SVR Base Model with MA Test vs Prediction')
plt.ylabel('close')
plt.xlabel('date')
plt.legend(['test', 'predicted'], loc='lower left')
plt.savefig(
    '/home/sklakshminarayanan/stock_data/model/svr_uv_ma_testvspred.png')
plt.show()

b = pd.DataFrame(yhat)
b['test'] = b[0]
c = pd.DataFrame(y_test)
b['Predicted'] = c[0]
a = pd.DataFrame(y_train)
a = a.append(b, ignore_index=True)
plt.plot(a[0])
plt.plot(a['Predicted'])
plt.plot(a['test'])
plt.title('SVR Base Model with MA Prediction')
plt.ylabel('stock price')
plt.xlabel('date')
plt.legend(['train', 'test', 'predicted'], loc='upper left')
plt.savefig('/home/sklakshminarayanan/stock_data/model/svr_uv_ma_predfull.png')
plt.show()
