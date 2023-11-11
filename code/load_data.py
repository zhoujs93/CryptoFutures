import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from ta import add_all_ta_features
import matplotlib.pyplot as plt

def plot_data():
    # df_all = pd.concat(dfs.values(), axis = 0, ignore_index = True)
    for k, v in dfs.items():
        ax = v.plot(x = 'close_time', y = 'close', figsize = (12, 12))
        plt.savefig(f'../data/plots/{k}.jpg')
        plt.show()
    return


if __name__ == '__main__':

    universe = ['BTCUSDT', 'BCHUSDT', 'BNBUSDT', 'BNBUSDT',
                'ETCUSDT', 'ETHUSDT', 'LTCUSDT', 'TRXUSDT',
                'XLMUSDT', 'ADAUSDT', 'IOTAUSDT', 'MKRUSDT',
                'DOGEUSDT', 'SOLUSDT']

    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
               'ignore']

    path = pathlib.Path.cwd().parent / 'data' / 'processed_futures'
    dfs = {}
    sample = {}
    for token in universe:
        name = f'{token}_1m.pickle'
        df = pd.read_pickle(str(path / name))
        df['open_time'] = pd.to_datetime(df['open_time'], unit = 'ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit = 'ms')
        df['token'] = token
        df = df.sort_values(by = 'open_time', ignore_index = True)
        df = add_all_ta_features(df, open = 'open', high = 'high', low = 'low',
                                 close = 'close', volume = 'volume', fillna = True)
        df['returns'] = df['close'].pct_change(periods=5)
        df['returns_5m'] = df['returns'].shift(-5)
        df.to_feather(f'../data/processed_data/{token}_1m.feather')

