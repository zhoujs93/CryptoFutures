import pandas as pd
import pathlib
from alpha101 import *
from factor_util import *
import matplotlib.pyplot as plt
import xarray as xr

def vwap(df):
    vol = df['volume']
    close = df['close']
    vwap_df = (vol * close).cumsum() / vol.cumsum()
    return vwap_df.values

def calculate_vwap(df):
    df['vwap'] = df.groupby(df['open_time'].dt.date, as_index = False).apply(vwap)
    return df

def plot_data():
    # df_all = pd.concat(dfs.values(), axis = 0, ignore_index = True)
    for k, v in dfs.items():
        ax = v.plot(x = 'close_time', y = 'close', figsize = (12, 12))
        plt.savefig(f'../data/plots/{k}.jpg')
        plt.show()
    return

def alpha001(close, returns):
    inner = close
    inner[returns < 0] = stddev(returns, 20)
    return ts_argmax(inner ** 2, 5).rank(axis=1, pct=True) - 0.5

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
    df_btc = calculate_vwap(df_btc)
    df_eth = calculate_vwap(df_eth)
    df = pd.concat([df_btc, df_eth, df_sol], axis = 0, ignore_index = True)
    df = df.sort_values(by = ['open_time'], ignore_index = True)
    df = df.iloc[:10000, :]

    df = df.set_index(['open_time', 'token'])
    df_xr = df.to_xarray()

    alpha = Alpha101(df_xr)
    alphas_all = {}

    alphas_all['alpha001'] = alpha.alpha001()
    alphas_all['alpha002'] = alpha.alpha002()

    # alphas_all['alpha003'] = alpha.alpha003()
    # alphas_all['alpha004'] = alpha.alpha004()
    #
    # # alphas_all['alpha005'] = alpha.alpha005()
    # alphas_all['alpha006'] = alpha.alpha006()
    # alphas_all['alpha007'] = alpha.alpha007()
    # alphas_all['alpha008'] = alpha.alpha008()
    # alphas_all['alpha009'] = alpha.alpha009()
    # alphas_all['alpha010'] = alpha.alpha010()
    # alphas_all['alpha011'] = alpha.alpha011()
    # alphas_all['alpha012'] = alpha.alpha012()
    # alphas_all['alpha013'] = alpha.alpha013()
    # alphas_all['alpha014'] = alpha.alpha014()
    # alphas_all['alpha015'] = alpha.alpha015()
    # alphas_all['alpha016'] = alpha.alpha016()
    # alphas_all['alpha017'] = alpha.alpha017()
    # alphas_all['alpha018'] = alpha.alpha018()
    # alphas_all['alpha019'] = alpha.alpha019()
    # alphas_all['alpha020'] = alpha.alpha020()
    # alphas_all['alpha021'] = alpha.alpha021()
    # # alpha022 = alpha.alpha022()
    # # alpha023 = alpha.alpha023()
    # alphas_all['alpha024'] = alpha.alpha024()
    # alphas_all['alpha025'] = alpha.alpha025()
    #
    # # alpha065 = alpha.alpha065()
    # # alpha066 = alpha.alpha066()
    # # alpha068 = alpha.alpha068()
    # # alpha071 = alpha.alpha071()
    # alphas_all['alpha072'] = alpha.alpha072()
    # # alpha073 = alpha.alpha073()
    # # alpha074 = alpha.alpha074()
    # # alpha075 = alpha.alpha075()
    # # alpha077 = alpha.alpha077()
    # alphas_all['alpha078'] = alpha.alpha078()
    # alphas_all['alpha081'] = alpha.alpha081()
    # alphas_all['alpha083'] = alpha.alpha083()
    # alphas_all['alpha084'] = alpha.alpha084()
    # alphas_all['alpha085'] = alpha.alpha085()
    # # alpha086 = alpha.alpha086()
    # # alphas_all['alpha088'] = alpha.alpha088()
    # # alpha092 = alpha.alpha092()
    # alphas_all['alpha094'] = alpha.alpha094()
    # # alpha095 = alpha.alpha095()
    # # alpha096 = alpha.alpha096()
    # # alpha098 = alpha.alpha098()
    # # alpha099 = alpha.alpha099()
    # alphas_all['alpha101'] = alpha.alpha101()