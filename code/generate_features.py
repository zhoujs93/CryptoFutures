import pandas as pd
import pathlib
from factor_util import *
import matplotlib.pyplot as plt
import seaborn as sns
import talib

def hullMA(x, n = 50):
    sma1 = x.rolling(n,  min_periods = 1).mean()
    sma2 = x.rolling(int(n/2),  min_periods = 1).mean()
    out = (2 * sma1 - sma2).rolling(int(np.sqrt(n)), min_periods = 1).mean()
    return x - out

def calculate_corr(df, ta_features = None, groupby = True):
    if ta_features is None:
        skip_features = ['returns_5m', 'open_time', 'close_time', 'target_15m', 'ignore', 'token']
        features = [x for x in df.columns if x not in skip_features]
        ta_features = [x for x in df.columns if x not in skip_features and x not in columns]
    if groupby:
        tgt_corr = df.groupby(['token'])[ta_features + ['target_15m']].corr()
    else:
        tgt_corr = df[ta_features + ['target_15m']].corr()
    return tgt_corr

def calculate_vol_price_corr(df, windows = [5, 15, 30, 60, 120]):
    for window in windows:
        df[f'vol_price_corr_{window}'] = df['close'].rolling(window, min_periods = 1).corr(df['volume'])
    return df

def get_cols_for_corr(df, str_idx):
    return df.columns[df.columns.str.startswith(str_idx)].tolist()

def transform_time(df):
    day = 24 * 60
    hour_float = df['open_time'].dt.hour + df['open_time'].dt.minute/60
    df['sin_hour'] = np.sin(2.0 * np.pi * hour_float/24)
    df['cos_hour'] = np.cos(2.0 * np.pi * hour_float/24)
    df['Day_sin'] = np.sin(df['open_time'].dt.day * (2 * np.pi / 31))
    df['Day_cos'] = np.cos(df['open_time'].dt.day * (2 * np.pi / 31))
    df['month_sin'] = np.sin(df['open_time'].dt.month * (2 * np.pi / 12))
    df['month_cos'] = np.cos(df['open_time'].dt.month * (2 * np.pi / 12))
    return df

def calc_sma_diff_test(close, timeperiod_short, timeperiod_long):
    res_short = close.rolling(window = timeperiod_short, min_periods = 1).mean()
    res_long = close.rolling(window = timeperiod_long, min_periods = 1).mean()
    res = (res_long - res_short) / res_long
    return res

def load_metrics_data(ticker):
    df_metrics = pd.read_feather(f'../data/processed_metrics/{ticker}_1m.feather')
    df_metrics['create_time'] = pd.to_datetime(df_metrics['create_time'], format = 'mixed')
    return df_metrics

if __name__ == '__main__':

    directory = '../data/processed_data/'
    files = pathlib.Path(directory).glob('*.feather')
    dfs = {}
    universe = ['BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'BNBUSDT',
                'ETCUSDT', 'ETHUSDT', 'LTCUSDT', 'TRXUSDT',
                'XLMUSDT', 'ADAUSDT', 'IOTAUSDT', 'MKRUSDT',
                'DOGEUSDT', 'SOLUSDT']

    df_btc = pd.read_feather('../data/processed_data/BTCUSDT_5m.feather')
    df = df_btc.copy()
    # df_eth = pd.read_feather('../data/processed_data/ETHUSDT_1m.feather')
    # df_sol = pd.read_feather('../data/processed_data/SOLUSDT_1m.feather')

    # metric_list = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    # metrics_df = [load_metrics_data(metric) for metric in metric_list]

    # df_metrics = pd.concat(metrics_df, axis = 0)
    # df_metrics['create_time'] = pd.to_datetime(df_metrics['create_time'], format = 'mixed')
    # metrics_columns = ['create_time', 'symbol', 'sum_open_interest', 'sum_open_interest_value',
    #                    'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio',
    #                    'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']
    #

    # df = pd.concat([df_btc, df_eth], axis = 0, ignore_index = True)
    df = df.sort_values(by = ['open_time'], ignore_index = True)
    # calculate next 15min returns (ie: current open_time is 2020-01-01 00:00:00,
    # then return is from 2020-01-01 00:01:00 - 2020-01-01 00:16:00
    df['target_15m'] = df.groupby(['token'])['close'].transform(lambda x: -1*x.pct_change(-15).shift(-1))


    # df = df.merge(df_metrics, left_on = ['open_time', 'token'], right_on = ['create_time', 'symbol'],
    #               how = 'left')
    #
    # df[metrics_columns] = df[metrics_columns].ffill()
    # df_sample = df.iloc[100000:110000, :]

    df_btc['returns_15m'] = df_btc['close'].pct_change(-15)
    df_btc['target_15m'] = -1 * df_btc['returns_15m'].shift(-1)
    df_btc['test_15m'] = (df_btc['close'].shift(-16) / df_btc['close'].shift(-1)) - 1
    df_btc['close_16'] = df_btc['close'].shift(-16)
    df_btc['close_1'] = df_btc['close'].shift(-1)
    df_btc['open_1'] = df_btc['open'].shift(-1)

    df_test = df_btc[['open_time', 'target_15m','close', 'close_16', 'close_1', 'open', 'open_1']]
    df_test = df_test.set_index(['open_time'])
    df_test = df_test.iloc[-10000:, :]

    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
               'ignore', 'target_15']

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

    # unless we have multiple tokens in our sample
    # df[f'mean_close/mean_{lag}'] = df.groupby(['open_time'])[f'log_close/mean_{lag}'].transform(lambda x: x.mean())
    # df[f'mean_log_returns_{lag}'] = df.groupby(['open_time'])[f'log_return_{lag}'].transform(lambda x: x.mean())
    #
    # df[f'log_close_mean_ratio_{lag}'] = df[f'log_close/mean_{lag}'] - df[f'mean_close/mean_{lag}']
    # df[f'log_return_{lag}_mean_log_returns_{lag}'] = df[f'log_return_{lag}'] - df[f'mean_log_returns_{lag}']

    df['mid_diff'] = (df['close'] - df['open']) / ((df['high'] - df['low']) + 0.001)

    df['target_return'] = token_grouped['close'].transform(lambda x: (x.shift(1) / x.shift(16)) - 1)

    sma_lags = [5, 15, 30, 60, 240, 800]
    grouped_df = df.groupby(['token'])
    for sma_lag in sma_lags:
        df[f'sma{sma_lag}'] = grouped_df['close'].transform(lambda x: (x.rolling(sma_lag, min_periods = 1).mean() / x) - 1)
        df[f'return{sma_lag}'] = grouped_df['close'].transform(lambda x: x.pct_change(sma_lag))
        df[f'volume_change_{sma_lag}'] = grouped_df['volume'].transform(lambda x: x.pct_change(sma_lag))

    hull_lags = [76, 240, 800]
    for hull_lag in hull_lags:
        df[f'hull_{hull_lag}'] = grouped_df['close'].transform(lambda x: hullMA(x, hull_lag))

    fibo_list = [55, 210, 340, 890, 3750]
    for num in fibo_list:
        df[f'log_return_{num}'] = grouped_df['close'].transform(lambda x: np.log(x).diff().rolling(num, min_periods = 1).mean().ffill().bfill())


    momentum_windows = [15, 30, 60, 120, 240]
    for window in momentum_windows:
        df_btc[f'mom_adx_{window}'] = talib.ADX(df_btc['high'], df_btc['low'], df_btc['close'], timeperiod = window)
        df_btc[f'mom_adxr_{window}'] = talib.ADXR(df_btc['high'], df_btc['low'], df_btc['close'], timeperiod=window)

    df = transform_time(df)

    sma_diff_windows = [(12, 26), (12*4*4, 24*4*4), (12*4*4*4, 24*4*4*4), (12*4*4*4*4, 24*4*4*4*4)]
    for short_win, long_win in sma_diff_windows:
        df[f'sma_diff_{short_win}'] = calc_sma_diff_test(df['close'], short_win, long_win)
    df[f'sma_diff_vol_{12*4*4}'] = calc_sma_diff_test(df['volume'], 12*4*4, 24*4*4)

    cols_to_drop = ['open_time', 'close_time', 'ignore',
                    'create_time', 'symbol', 'returns', 'returns_5m',
                    'open', 'high', 'low', 'close']

    cols = pd.DataFrame(df.columns)
    df.to_feather('../data/df_btc_with_features_5m.feather')
    # SMA, WMA, RSI, ROC, Mo, OBV, log return, max in range, min in range,
    # middle, EMA, MSTD, MVAR, RSV, RSI, KDJ, Boll,
    # MACD, CR, WR, CCI, TR, ATR, DMA< DMI, DI< ADX, ADXR, TRIX, TEMA, VR
