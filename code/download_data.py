import zipfile, shutil, pathlib, io, requests, glob, os, sys
import pandas as pd
import subprocess, time

def convert_to_single_file(write_dir, files, time, symbol):
    dfs = []
    errors = []
    for file in files:
        try:
            df = pd.read_csv(str(file), header=None)
            dfs.append(df)
        except Exception as e:
            print(f'Error during loading: {e}')
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    df_final = []
    header = None
    for idx, row in df_all.iterrows():
        if row[0] != 'open_time':
            df_final.append(row)
        else:
            header = row.tolist()
    df_filter = pd.DataFrame(df_final)
    df_filter.columns = header
    df_filter['symbol'] = symbol
    df_filter['open_time'] = pd.to_datetime(df_filter['open_time'], unit = 'ms')
    # data_dir = pathlib.Path.cwd() / 'data' / symbol
    filename = f'{symbol}-{time}.csv'
    df_filter = df_filter.sort_values(by = ['open_time'], ignore_index = True)
    df_filter.to_csv(str(write_dir / filename))
    return df_filter

def download_data(universe):
    command = f'python3 download-kline.py -t um -i 1m -y 2020 2021 2022 -s {universe} -folder ~/CandyMachineV2/CryptoTrading/binance_data -skip-monthly 1'
    binance_directory = pathlib.Path.cwd().parent / 'binance-public-data' / 'python'
    commands = f'{command}'
    process = subprocess.Popen(commands, cwd = str(binance_directory), shell = True,
                   stdin=subprocess.PIPE, stdout=sys.stdout)
    grep_stdout = process.communicate(input='n'.encode())[0]
    return None

if __name__ == '__main__':
    data = pd.read_csv('./data/universe.csv')
    end_date = '2022-09-15'
    times = ['1d']
    data_dir = pathlib.Path.cwd() / 'binance_data'
    df = pd.read_pickle(str(data_dir / 'data_universe.pickle'))
    numeric = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
               'taker_buy_volume', 'taker_buy_quote_volume']
    symbol = data['Symbol']
    symbols = symbol.tolist()
    universes = []
    universes = [x for x in symbols if 'BUSD' not in x]


    # idx = universes.index('TLMUSDT')
    # universes = universes[idx:]

    # path = './binance_data/data/futures/um/daily/klines/'
    # for universe in universes:
    #     btc_dir = pathlib.Path.cwd() / 'binance_data' / 'data' / 'futures' / 'um' / 'daily' / 'klines' / universe / '1m'
    #     x = list(btc_dir.glob('*.zip'))
    #     complete = False
    #     if len(x) == 0:
    #         print(x)
    #         while complete == False:
    #             try:
    #                 download_data(universe)
    #                 complete = True
    #             except Exception as e:
    #                 print(e)
    #                 time.sleep(10)

    # download data
    # subfolders = os.listdir(path)
    # for char in string.ascii_uppercase[3:]:
    #     universe = ' '.join([x for x in symbols if x not in subfolders and x.startswith(char)])
    #     download_data(data, universe)

    # for time in times:
    #     for i, row in data.iterrows():
    #         start_date = row['Start Date']
    #         dates = pd.date_range(start = start_date, end = end_date).tolist()
    #         symbol = row['Symbol']
    #         write_dir = data_dir / symbol / time
    #         write_dir.mkdir(parents = True, exist_ok = True)
    #         errors = []
    #         for index, date in enumerate(dates):
    #             if index % 10 == 0:
    #                 print(symbol, date)
    #             t = date.strftime('%Y-%m-%d')
    #             try:
    #                 url = f'https://data.binance.vision/data/futures/um/daily/klines/{symbol}/{time}/{symbol}-{time}-{t}.zip'
    #                 file_name = f'{t}.csv'
    #                 r = requests.get(url)
    #                 z = zipfile.ZipFile(io.BytesIO(r.content))
    #                 z.extractall(str(write_dir))
    #             except Exception as e:
    #                 print(f'Error with {t}: {e}')
    #                 errors.append(t)
    #         files = write_dir.glob('*.csv')
    #         df = convert_to_single_file(files, time, symbol)

    path = './binance_data/data/futures/um/daily/klines/'
    data_dir = pathlib.Path.cwd() / 'binance_data'/ 'data' / 'futures' / 'um' / 'daily' / 'klines'
    subfolders = os.listdir(path)
    time = '1m'
    dfs_merged = {}
    errors = []
    for symbol in subfolders:
        if symbol in universes and symbol not in dfs_merged:
            write_dir = data_dir / symbol / time
            save_dir = data_dir / symbol
            file_path = save_dir / f'{symbol}-1m.csv'
            if write_dir.is_dir() and (file_path.exists()):
                print(write_dir)
                df = pd.read_csv(str(file_path), header = 0)
                dfs_merged[symbol] = df
                # files = write_dir.glob('*.zip')
                # try:
                #     for i, file in enumerate(files):
                #         if i == 0:
                #             print(file)
                #
                        # with zipfile.ZipFile(file, "r") as zip_ref:
                        #     zip_ref.extractall(str(write_dir))
                    # files = write_dir.glob('*.csv')
                    # df = convert_to_single_file(save_dir, files, time, symbol)
                    # dfs_merged[symbol] = df
                # except Exception as e:
                #     print(f'Error for {symbol} : {e}')
                #     errors.append(symbol)

    dfs = pd.concat(dfs_merged, axis = 0, ignore_index = True)
    dfs['open_time'] = pd.to_datetime(dfs['open_time'])
    dfs['close_time'] = pd.to_datetime(dfs['close_time'])
    dfs.to_pickle('./binance_data/data_universe_1m.pickle')
    # for k, v in dfs_merged.items():
    #     try:
    #         dfs_merged[k]['open_time'] = pd.to_datetime(v['open_time'])
    #     except Exception as e:
    #         print(f'{k}: {e}')
    # dfs['day_of_year'] = dfs['open_time'].dt.dayofyear
    # dfs['ordinal_date'] = dfs['open_time'].apply(lambda x: x.toordinal())
    # dfs['year'] = dfs['open_time'].dt.year
    # years = dfs['year'].unique()
    # for year in years:
    #     temp_df = dfs.loc[(dfs['year'] == year), :]

