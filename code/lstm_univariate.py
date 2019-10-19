import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

data_raw = pd.read_csv('/home/sklakshminarayanan/Downloads/DJI.csv', index_col='Date')
data = data_raw.drop('Close', axis=1)
data['Close'] = data_raw['Close']
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
model.add(LSTM(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='SGD')


history = model.fit(train_X, train_y, epochs=100, batch_size=4, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
print(history.history)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LSTM Base Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('/home/sklakshminarayanan/stock_data/model/lstm_uv_loss.png')
plt.show()


yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], 5))
temp = test_X[:, -5:]

inv_yhat = np.concatenate((yhat, temp), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -5:]), axis=1)
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
loss=pd.DataFrame(loss,index=[0])
loss.to_csv('/home/sklakshminarayanan/stock_data/data/lstm_uv_loss.csv',index=None)

lstm1 = {}
lstm1['pred'] = inv_yhat
lstm1['test'] = inv_y
lstm_uv = pd.DataFrame(lstm1)
lstm_uv.to_csv('/home/sklakshminarayanan/stock_data/data/lstm_uv_pred.csv',index=None)

plt.plot(inv_y)
plt.plot(inv_yhat)
plt.title('LSTM Base Model test vs prediction')
plt.ylabel('close')
plt.xlabel('date')
plt.legend(['test', 'predicted'], loc='lower left')
plt.savefig('/home/sklakshminarayanan/stock_data/model/lstm_uv_testvspred.png')
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
plt.title('LSTM Base Model Prediction')
plt.ylabel('stock price')
plt.xlabel('date')
plt.legend(['train', 'test', 'predicted'], loc='upper left')
plt.savefig('/home/sklakshminarayanan/stock_data/model/lstm_uv_predfull.png')
plt.show()
