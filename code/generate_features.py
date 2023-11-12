import pandas as pd
import pathlib
from factor_util import *
import matplotlib.pyplot as plt
import plotly.express as px

def hullMA(x, n = 50):
    sma1 = x.rolling(n).mean()
    sma2 = x.rolling(int(n/2)).mean()
    out = (2 * sma1 - sma2).rolling(int(np.sqrt(n))).mean()
    return x - out

def calculate_corr(df):
    skip_features = ['returns_5m', 'open_time', 'close_time', 'target_15m', 'ignore', 'token']
    features = [x for x in df.columns if x not in skip_features]
    ta_features = [x for x in df.columns if x not in skip_features and x not in columns]
    tgt_corr = df.groupby(['token'])[ta_features + ['target_15m']].corr()
    return tgt_corr

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
    # df_sol = pd.read_feather('../data/processed_data/SOLUSDT_1m.feather')

    df = pd.concat([df_btc, df_eth], axis = 0, ignore_index = True)
    df = df.sort_values(by = ['open_time'], ignore_index = True)
    df['target_15m'] = -1 * df.groupby(['token'])['close'].pct_change(-15).shift(-1)
    df = df.iloc[1000000:1100000, :]


    df_btc['returns_15m'] = df_btc['close'].pct_change(-15)
    df_btc['fwd_returns_15m'] = -1 * df_btc['returns_15m'].shift(-1)
    df_btc['test_15m'] = (df_btc['close'].shift(-16) / df_btc['close'].shift(-1)) - 1
    df_btc['close_16'] = df_btc['close'].shift(-16)
    df_btc['close_1'] = df_btc['close'].shift(-1)
    df_btc['open_1'] = df_btc['open'].shift(-1)

    df_test = df_btc[['open_time', 'fwd_returns_15m','close', 'close_16', 'close_1', 'open', 'open_1']]
    df_test = df_test.set_index(['open_time'])
    df_test = df_test.iloc[-10000:, :]

    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
               'ignore']

    token_grouped = df.groupby(['token'])

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
    df[f'log_close/mean_{lag}'] = token_grouped['close'].transform(lambda x: gen_cross_features(x, lag = lag))
    df[f'log_return_{lag}'] = token_grouped['close'].transform(lambda x: log_return_np(x))

    df[f'mean_close/mean_{lag}'] = df.groupby(['open_time'])[f'log_close/mean_{lag}'].transform(lambda x: x.mean())
    df[f'mean_log_returns_{lag}'] = df.groupby(['open_time'])[f'log_return_{lag}'].transform(lambda x: x.mean())

    df[f'log_close_mean_ratio_{lag}'] = df[f'log_close/mean_{lag}'] - df[f'mean_close/mean_{lag}']
    df[f'log_return_{lag}_mean_log_returns_{lag}'] = df[f'log_return_{lag}'] - df[f'mean_log_returns_{lag}']



    df['target_return'] = token_grouped['close'].transform(lambda x: (x.shift(1) / x.shift(16)) - 1)

    sma_lags = [15, 60, 240]
    grouped_df = df.groupby(['token'])
    for sma_lag in sma_lags:
        df[f'sma{sma_lag}'] = grouped_df['close'].transform(lambda x: (x.rolling(lag).mean() / x) - 1)
        df[f'return{sma_lag}'] = grouped_df['close'].transform(lambda x: x.pct_change(sma_lag))
        df[f'volume_change_{sma_lag}'] = grouped_df['volume'].transform(lambda x: x.pct_change(sma_lag))

    hull_lags = [76, 240, 800]
    for hull_lag in hull_lags:
        df[f'hull_{hull_lag}'] = grouped_df['close'].transform(lambda x: hullMA(x, hull_lag))

    fibo_list = [55, 210, 340, 890, 3750]
    for num in fibo_list:
        df[f'log_return_{num}'] = grouped_df['close'].transform(lambda x: np.log(x).diff().rolling(num).mean().ffill().bfill())

    corr_df = calculate_corr(df)

    ax = df.loc[df['token'] == 'BTCUSDT']['target_15m'].hist(figsize = (12,12), bins = 100)
    plt.show()

    fig = px.histogram(df, x = 'target_15m', color = '')