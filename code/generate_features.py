import pandas as pd
import pathlib
from alpha101 import *
from factor_util import *
import matplotlib.pyplot as plt

def calcHullMA(price: pd.Series, N=50):
    SMA1 = price.rolling(N).mean()
    SMA2 = price.rolling(int(N/2)).mean()
    return (2 * SMA2 - SMA1).rolling(int(np.sqrt(N))).mean()


if __name__ == '__main__':

    directory = '../data/processed_data/'
    files = pathlib.Path(directory).glob('*.feather')
    dfs = {}
    universe = ['BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'BNBUSDT',
                'ETCUSDT', 'ETHUSDT', 'LTCUSDT', 'TRXUSDT',
                'XLMUSDT', 'ADAUSDT', 'IOTAUSDT', 'MKRUSDT',
                'DOGEUSDT', 'SOLUSDT']

    df_btc = pd.read_feather('../data/processed_data/BTCUSDT_1m.feather')
    df_eth = pd.read_feather('../data/processed_data/ETHUSDT_1m.feather')
    df_sol = pd.read_feather('../data/processed_data/SOLUSDT_1m.feather')

    df = pd.concat([df_btc, df_eth, df_sol], axis = 0, ignore_index = True)
    df = df.sort_values(by = ['open_time'], ignore_index = True)
    # df = df.iloc[1000000:1100000, :]


    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
               'ignore']

    df['returns_15m'] = df.groupby(['token'])['close'].pct_change(15)
    df['target_15m'] = df.groupby(['token'])['returns_15m'].shift(-1)

    def gen_cross_features(x, lag = 60):
        lag_arr = np.ones(lag)
        conv_arr = np.convolve(x, lag_arr / lag, mode = 'valid')
        app_arr = np.append(conv_arr, np.ones(lag - 1))
        roll_arr = np.roll(app_arr, lag - 1)
        div_arr = np.log(x / roll_arr)
        return div_arr

    def log_return_np(x):
        return np.log(x / x.shift(60)).fillna(0)

    lag = 60
    df[f'log_close/mean_{lag}'] = df.groupby(['token'])['close'].transform(lambda x: gen_cross_features(x, lag = lag))
    df[f'log_return_{lag}'] = df.groupby(['token'])['close'].transform(lambda x: log_return_np(x))

    df[f'mean_close/mean_{lag}'] = df.groupby(['open_time'])[f'log_close/mean_{lag}'].transform(lambda x: x.mean())
    df[f'mean_log_returns_{lag}'] = df.groupby(['open_time'])[f'log_return_{lag}'].transform(lambda x: x.mean())

    df[f'log_close_mean_ratio_{lag}'] = df[f'log_close/mean_{lag}'] - df[f'mean_close/mean_{lag}']
    df[f'log_return_{lag}_mean_log_returns_{lag}'] = df[f'log_return_{lag}'] - df[f'mean_log_returns_{lag}']

    skip_features = ['returns_5m', 'open_time', 'close_time', 'target_15m', 'ignore', 'token']
    features = [x for x in df.columns if x not in skip_features]
    ta_features = [x for x in df.columns if x not in skip_features and x not in columns]
    tgt_corr = df.groupby(['token'])[ta_features + ['target_15m']].corr()
    t = tgt_corr['target_15m']