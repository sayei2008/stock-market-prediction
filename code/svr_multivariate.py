import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
a = pd.read_csv('/home/sklakshminarayanan/stock_data/crudeoil.csv', index_col='Date')
b = pd.read_csv('/home/sklakshminarayanan/stock_data/gold.csv', index_col='Date')
a = pd.merge(a, b, on='Date', how='inner')
data_raw = pd.read_csv('/home/sklakshminarayanan/stock_data/stock.csv', index_col='Date')
data_raw = pd.merge(data_raw, a, on='Date', how='inner')
data = data_raw
start_train = '2015-01-01'
end_train = '2018-01-05'
start_test = '2018-01-06'
end_test = '2018-12-31'
data_train = data.ix[start_train:end_train]
X_train = data_train.drop('Close', axis=1).values
X_train = data_train.drop('Adj Close', axis=1).values
y_train = data_train['Close'].values
data_test = data.ix[start_test:end_test]
X_test = data_test.drop('Close', axis=1).values
X_test = data_test.drop('Adj Close', axis=1).values
y_test = data_test['Close'].values
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)
svr = SVR(kernel='linear')
svr.fit(X_scaled_train,y_train)
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
loss=pd.DataFrame(loss,index=[0])
loss.to_csv('/home/sklakshminarayanan/stock_data/data/svr_mv_loss.csv',index=None)

svr1 = {}
svr1['pred'] = yhat
svr1['test'] = y_test
svr_mv = pd.DataFrame(svr1)
svr_mv.to_csv('/home/sklakshminarayanan/stock_data/data/svr_mv_pred.csv',index=None)

plt.plot(y_test)
plt.plot(yhat)
plt.title('SVR Advanced Model Test vs Prediction')
plt.ylabel('close')
plt.xlabel('date')
plt.legend(['test', 'predicted'], loc='lower left')
plt.savefig('/home/sklakshminarayanan/stock_data/model/svr_mv_testvspred.png')
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
plt.title('SVR Advanced Model Prediction')
plt.ylabel('stock price')
plt.xlabel('date')
plt.legend(['train', 'test', 'predicted'], loc='upper left')
plt.savefig('/home/sklakshminarayanan/stock_data/model/svr_mv_predfull.png')
plt.show()

