import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from ta import add_all_ta_features
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_data(dfs):
    # df_all = pd.concat(dfs.values(), axis = 0, ignore_index = True)
    for k, v in dfs.items():
        ax = v.plot(x = 'close_time', y = 'close', figsize = (12, 12))
        plt.savefig(f'../data/plots/{k}.jpg')
        plt.show()
    return

def process_klines_data(columns, interval, ticker):
    path = pathlib.Path.cwd().parent / 'data' / "futures" / 'um' / 'daily' / 'klines'
    token_lists = [f.name for f in path.iterdir()]
    for token in token_lists:
        try:
            files = path / token / f'{interval}'
            csv_files = files.glob('*.csv')
            dfs = []
            for file in csv_files:
                df = pd.read_csv(file, index_col=None, names=columns)
                idx = df.index[df['open_time'] != 'open_time']
                df = df.loc[idx].reset_index(drop=True)
                for col in columns:
                    df[col] = df[col].astype(float)
                dfs.append(df)
            df_all = pd.concat(dfs, axis=0, ignore_index=True)
            # write_dir = pathlib.Path.cwd().parent / 'data' / 'processed_futures'
            # write_dir.mkdir(parents=True, exist_ok=True)
            # file_name = f'{token}_1m.pickle'
            # df_all.to_pickle(str(write_dir / file_name))
        except Exception as e:
            print(f'Error for {token}: {e}')
    return df_all

def load_klines_data(universe, columns, interval = '5m', save = True):
    dfs = {}
    sample = {}
    for token in tqdm(universe):
        df = process_klines_data(columns, interval = interval, ticker = token)
        df['open_time'] = pd.to_datetime(df['open_time'], unit = 'ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit = 'ms')
        df_prev = pd.read_feather(f'../data/processed_futures/BTCUSDT_5m.feather')
        df_prev = df_prev[columns]
        df = pd.concat([df_prev, df], axis = 0)
        print(f'Shape of df is {df.shape}')
        df = df.drop_duplicates(subset = ['open_time'])
        print(f'Shape of df after is {df.shape}')
        df['token'] = token
        df = df.sort_values(by = 'open_time', ignore_index = True)
        df = add_all_ta_features(df, open = 'open', high = 'high', low = 'low',
                                 close = 'close', volume = 'volume', fillna = True)
        df['target_15m'] = -1 * df['close'].pct_change(-15).shift(-1)
        if save:
            df.to_feather(f'../data/processed_data/{token}_{interval}.feather')
        dfs[token] = df
    return dfs

if __name__ == '__main__':

    # universe = ['BTCUSDT', 'BCHUSDT', 'BNBUSDT', 'BNBUSDT',
    #             'ETCUSDT', 'ETHUSDT', 'LTCUSDT', 'TRXUSDT',
    #             'XLMUSDT', 'ADAUSDT', 'IOTAUSDT', 'MKRUSDT',
    #             'DOGEUSDT', 'SOLUSDT']

    universe = ['BTCUSDT']
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
               'ignore']

    path = pathlib.Path.cwd().parent / 'data' / 'processed_futures'
    dfs = load_klines_data(universe, columns, interval = '5m')