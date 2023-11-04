import pandas as pd
import pathlib

if __name__ == '__main__':
    universe = ['BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'EOSUSDT',
                'ETCUSDT', 'ETHUSDT', 'LTCUSDT', 'TRXUSDT',
                'XLMUSDT', 'ADAUSDT', 'IOTAUSDT', 'MKRUSDT',
                'DOGEUSDT']

    lags = [60, 300, 900]
    path = pathlib.Path.cwd().parent / 'data' / 'processed_futures'
    dfs = {}
    for token in universe:
        name = f'{token}_1m.feather'
        df = pd.read_feather(str(path / name))
        dfs[token] = df