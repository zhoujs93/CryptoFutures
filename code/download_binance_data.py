from binance_historical_data import BinanceDataDumper
import pandas as pd
import pathlib
import glob

if __name__ == '__main__':
    data_dumper = BinanceDataDumper(
        path_dir_where_to_dump = '../data/',
        asset_class = 'um',
        data_type = 'klines',
        data_frequency = '1m'
    )
    x = data_dumper.dump_data(
        tickers = None,
        date_start = None,
        date_end = None,
        is_to_update_existing = False,
        tickers_to_exclude = ["UST"]
    )
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
               'ignore']
    path = pathlib.Path.cwd().parent / 'data' / "futures" / 'um' / 'monthly' / 'klines'
    token_lists = [f.name for f in path.iterdir()]
    for token in token_lists:
        try:
            files = path / token / '1m'
            csv_files = files.glob('*.csv')
            dfs = []
            for file in csv_files:
                df = pd.read_csv(file, index_col = None, names = columns)
                idx = df.index[df['open_time'] != 'open_time']
                df = df.loc[idx].reset_index(drop = True)
                for col in columns:
                    df[col] = df[col].astype(float)
                dfs.append(df)
            df_all = pd.concat(dfs, axis = 0, ignore_index = True)
            write_dir = pathlib.Path.cwd().parent / 'data' / 'processed_futures'
            write_dir.mkdir(parents = True, exist_ok = True)
            file_name = f'{token}_1m.pickle'
            df_all.to_pickle(str(write_dir / file_name))
        except Exception as e:
            print(f'Error for {token}: {e}')

