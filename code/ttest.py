import numpy as np
import pandas as pd
from scipy import stats


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/lstm_uv_pred.csv')
print("LSTM Model 1 MAPE")
print(mean_absolute_percentage_error(a['test'], a['pred']))
a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/lstm_uv_ma_pred.csv')
print("LSTM Model 2 MAPE")
print(mean_absolute_percentage_error(a['test'], a['pred']))
a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/lstm_mv_pred.csv')
print("LSTM Model 3 MAPE")
print(mean_absolute_percentage_error(a['test'], a['pred']))
a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/lstm_mv_ma_pred.csv')
print("LSTM Model 4 MAPE")
print(mean_absolute_percentage_error(a['test'], a['pred']))
a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/svr_uv_pred.csv')
print("SVR Model 1 MAPE")
print(mean_absolute_percentage_error(a['test'], a['pred']))
a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/svr_uv_ma_pred.csv')
print("SVR Model 2 MAPE")
print(mean_absolute_percentage_error(a['test'], a['pred']))
a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/svr_mv_pred.csv')
print("SVR Model 3 MAPE")
print(mean_absolute_percentage_error(a['test'], a['pred']))
a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/svr_mv_ma_pred.csv')
print("SVR Model 4 MAPE")
print(mean_absolute_percentage_error(a['test'], a['pred']))

a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/lstm_uv_pred.csv')
b = pd.read_csv('/home/sklakshminarayanan/stock_data/data/lstm_mv_ma_pred.csv')
c = pd.merge(a, b, on='test', how='inner')
print("LSTM Model 1 vs 4 t-test")
print(stats.ttest_rel(c['pred_x'], c['pred_y']))
print(a.describe())
print(b.describe())

a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/lstm_uv_ma_pred.csv')
b = pd.read_csv('/home/sklakshminarayanan/stock_data/data/lstm_mv_ma_pred.csv')
c = pd.merge(a, b, on='test', how='inner')
print("LSTM Model 2 vs 4 t-test")
print(stats.ttest_rel(c['pred_x'], c['pred_y']))
print(a.describe())
print(b.describe())

a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/lstm_mv_pred.csv')
b = pd.read_csv('/home/sklakshminarayanan/stock_data/data/lstm_mv_ma_pred.csv')
c = pd.merge(a, b, on='test', how='inner')
print("LSTM Model 3 vs 4 t-test")
print(stats.ttest_rel(c['pred_x'], c['pred_y']))
print(a.describe())
print(b.describe())

a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/svr_uv_pred.csv')
b = pd.read_csv('/home/sklakshminarayanan/stock_data/data/lstm_mv_ma_pred.csv')
print("SVR model 1 vs LSTM model 4 t-test")
print(stats.ttest_ind(a['pred'],b['pred']))
print(a.describe())
print(b.describe())

a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/svr_uv_ma_pred.csv')
b = pd.read_csv('/home/sklakshminarayanan/stock_data/data/lstm_mv_ma_pred.csv')
print("SVR model 2 vs LSTM model 4 t-test")
print(stats.ttest_ind(a['pred'],b['pred']))
print(a.describe())
print(b.describe())

a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/svr_mv_pred.csv')
b = pd.read_csv('/home/sklakshminarayanan/stock_data/data/lstm_mv_ma_pred.csv')
print("SVR model 3 vs LSTM model 4 t-test")
print(stats.ttest_ind(a['pred'],b['pred']))
print(a.describe())
print(b.describe())

a = pd.read_csv('/home/sklakshminarayanan/stock_data/data/svr_mv_ma_pred.csv')
b = pd.read_csv('/home/sklakshminarayanan/stock_data/data/lstm_mv_ma_pred.csv')
print("SVR model 4 vs LSTM model 4 t-test")
print(stats.ttest_ind(a['pred'],b['pred']))
print(a.describe())
print(b.describe())

