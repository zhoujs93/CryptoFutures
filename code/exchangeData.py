import pandas as pd
from datetime import datetime, timedelta
from binance.enums import HistoricalKlinesType
import requests
import json
from mexc_sdk import Spot
import pandas as pd
from binance.cm_futures import CMFutures
import logging
from binance.um_futures import UMFutures
from binance.lib.utils import config_logging
import time

exchange_key = 'HHwPdDKvCO3zxbLqQbjxcb0N2wDNDs2aD5aIz3M3GsuiIntsgfV0wWWIaqfnKriw'
end_time = round(time.time() * 1000)
end_datetime = datetime.fromtimestamp(end_time / 1000.0)
config_logging(logging, logging.DEBUG)

um_futures_client = UMFutures(key = exchange_key)

kline = um_futures_client.klines("BTCUSDT", "1m", end_time = end_time)

columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
           'close_time', 'quote_asset_volume', 'number_of_trades',
           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
           'ignore']

df = pd.DataFrame(kline, columns = columns)
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

# logging.info(um_futures_client.klines("BTCUSDT", "1d"))


# client = Client(exchange_key)
# date_format = "%Y-%m-%d %H:%M:%S"
# start = datetime(2022, 1, 1, 16, 0, 0)
# end = datetime.now()
#
# mexc_api = 'mx0vglWv2zvTJs6wsG'
# mexc = '5fd10275f5064687adfb085ce060197b'
#
# klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE,
#                                       "1 day ago UTC", klines_type=HistoricalKlinesType.FUTURES)
#
# columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
#            'close_time', 'quote_asset_volume', 'number_of_trades',
#            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
#            'ignore']
# df = pd.DataFrame(klines, columns = columns)
# df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
# df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')


