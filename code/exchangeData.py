import pandas as pd
from binance import Client
from datetime import datetime, timedelta
import requests
import json

client = Client()
date_format = "%Y-%m-%d %H:%M:%S"
start = datetime(2022, 1, 1, 16, 0, 0)
end = datetime.now()


def get_binance_data(underlying: str, start: str, end: str, interval=client.KLINE_INTERVAL_1DAY):
    result = client.get_historical_klines(symbol=f'{underlying.upper()}USDT',
                                          start_str=start, end_str=end, interval=interval)

    closing_prices = [float(x[4]) for x in result]
    open_prices = [float(x[1]) for x in result]
    high_prices = [float(x[2]) for x in result]
    low_prices = [float(x[3]) for x in result]
    dates = [datetime.fromtimestamp(x[0] / 1000) for x in result]
    volume = [float(x[5]) for x in result]
    num_trades = [float(x[8]) for x in result]

    data = pd.DataFrame(index=dates)
    data['Open'] = open_prices
    data['High'] = high_prices
    data['Low'] = low_prices
    data['Close'] = closing_prices
    data['Volume'] = volume
    data['Number of Trades'] = num_trades

    return data


binance_data = get_binance_data("BTC", start.strftime(date_format), end.strftime(date_format))


def get_ftx_data(underlying: str, start: datetime, end: datetime, resolution=86400):
    FTX_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S+00:00"
    session = requests.session()
    api_url = "https://ftx.com/api"
    start = int(start.timestamp())
    end = int(end.timestamp())
    endpoint = f'/markets/{underlying}/candles?resolution={resolution}&start_time={start}&end_time={end}'

    response = session.get(api_url + endpoint)
    response = json.loads(response.text.encode('ascii'))

    if response['success']:
        data = response['result']
        closing_prices = [x['close'] for x in data]
        open_prices = [x['open'] for x in data]
        high_prices = [x['high'] for x in data]
        low_prices = [x['low'] for x in data]
        dates = [datetime.strptime(x['startTime'], FTX_DATE_FORMAT) for x in data]
        volume = [x['volume'] for x in data]

        ftx_data = pd.DataFrame(index=dates)
        ftx_data['Open'] = open_prices
        ftx_data['High'] = high_prices
        ftx_data['Low'] = low_prices
        ftx_data['Close'] = closing_prices
        ftx_data['Volume'] = volume

        return ftx_data

    else:
        raise Exception("no successful api response")


# ftx_data = get_ftx_data("BTC/USD", start, end)
