import pandas as pd
import requests
import json
import matplotlib.pyplot as plt


def fetch_daily_data(symbol):
    pair_split = symbol.split('/')  # must split in format XXX/XXX
    symbol = pair_split[0] + '-' + pair_split[1]
    url = f'https://api.pro.coinbase.com/products/{symbol}/candles?granularity=86400'
    response = requests.get(url)
    if response.status_code == 200:
        data = pd.DataFrame(json.loads(response.text), columns=['unix', 'low', 'high', 'open', 'close', 'volume'])
        data['date'] = pd.to_datetime(data['unix'], unit='s')
        data['vol_fiat'] = data['volume'] * data['close']
        if data is None:
            print("Did not get any data back from Coinbase for this symbol")
        else:
            data.to_csv(f'Coinbasedata/{pair_split[0] + pair_split[1]}_dailydata.csv', index=False)

    else:
        print("Did not receieve OK response from Coinbase API")
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    plt.figure(figsize=(15, 5))
    plt.title('{} / {} price data'.format(pair_split[0], pair_split[1]))
    plt.plot(data.date, data.close)
    plt.xlabel("Date")
    plt.ylabel('Close Price ({})'.format(pair_split[1]))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    pair = "ETH/USD"
    fetch_daily_data(symbol=pair)
