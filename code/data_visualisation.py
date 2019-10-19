import pandas as pd
import matplotlib.pyplot as plt


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
print(data_raw.describe())
st = data_raw.describe()
st.to_csv('/home/sklakshminarayanan/stock_data/data/describe.csv')
data = generate_features(data_raw)
data.index = pd.RangeIndex(len(data.index))
st = data.describe()
st.to_csv('/home/sklakshminarayanan/stock_data/data/describe_ma.csv')
print(data.shape)

plt.plot(data['close'])
plt.ylabel('Close')
plt.xlabel('Date')
plt.title('Close Stock Prices')
plt.savefig('/home/sklakshminarayanan/stock_data/close.png')
plt.show()

plt.plot(data['close'])
plt.plot(data['avg_price_5'])
plt.plot(data['avg_price_30'])
plt.plot(data['avg_price_365'])
plt.title('Close Stock Prices')
plt.ylabel('Close')
plt.xlabel('Date')
plt.legend(['Close', 'Average Price of 5 days', 'Average Price of 30 days',
            'Average Price of 365 days'],
           loc='upper left')
plt.savefig('/home/sklakshminarayanan/stock_data/close_ma.png')
plt.show()

plt.plot(data['volume_1'])
plt.title('volume Stock volumes')
plt.ylabel('volume')
plt.xlabel('Date')
plt.savefig('/home/sklakshminarayanan/stock_data/volume.png')
plt.show()

plt.plot(data['volume_1'])
plt.plot(data['avg_volume_5'])
plt.plot(data['avg_volume_30'])
plt.plot(data['avg_volume_365'])
plt.title('volume Stock volumes')
plt.ylabel('volume')
plt.xlabel('Date')
plt.legend(['volume', 'Average volume of 5 days', 'Average volume of 30 days',
            'Average volume of 365 days'],
           loc='upper left')
plt.savefig('/home/sklakshminarayanan/stock_data/volume_ma.png')
plt.show()

plt.plot(data['crudeoil_1'])
plt.title('crudeoil cost')
plt.ylabel('crudeoil')
plt.xlabel('Date')
plt.savefig('/home/sklakshminarayanan/stock_data/crude.png')

plt.plot(data['crudeoil_1'])
plt.plot(data['avg_crudeoil_5'])
plt.plot(data['avg_crudeoil_30'])
plt.plot(data['avg_crudeoil_365'])
plt.title('crudeoil cost')
plt.ylabel('crudeoil')
plt.xlabel('Date')
plt.legend(
    ['crudeoil', 'Average crudeoil of 5 days', 'Average crudeoil of 30 days',
     'Average crudeoil of 365 days'],
    loc='upper left')
plt.savefig('/home/sklakshminarayanan/stock_data/crude_ma.png')
plt.show()

plt.plot(data['gold_1'])
plt.title('gold cost')
plt.ylabel('gold')
plt.xlabel('Date')
plt.savefig('/home/sklakshminarayanan/stock_data/gold.png')
plt.show()

plt.plot(data['gold_1'])
plt.plot(data['avg_gold_5'])
plt.plot(data['avg_gold_30'])
plt.plot(data['avg_gold_365'])
plt.title('gold cost')
plt.ylabel('gold')
plt.xlabel('Date')
plt.legend(['gold', 'Average gold of 5 days', 'Average gold of 30 days',
            'Average gold of 365 days'],
           loc='upper left')
plt.savefig('/home/sklakshminarayanan/stock_data/gold_ma.png')
plt.show()

data['Daily Return'] = data['close'].pct_change()
plt.plot(data['Daily Return'], 'bo--', linewidth=1, markersize=2)
plt.title('Daily Returns for the stock')
plt.legend(['daily return'])
plt.savefig('/home/sklakshminarayanan/stock_data/returns.png')
plt.show()

plt.hist(data['close'])
plt.title("DJI Stock Data Histogram")
plt.savefig('/home/sklakshminarayanan/stock_data/histogram.png')
plt.show()
