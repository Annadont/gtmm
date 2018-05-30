import requests
import pandas as pd
from datetime import datetime as dt


class CurrentPrice:

    def __init__(self, data_path):
        self.data = pd.DataFrame.from_csv(path=data_path, sep=",")
        self.data = self.data.drop(columns=["Open", "High", "Low", "Close", "Volume_(BTC)", "Volume_(Currency)"])
        self.data = self.data.reset_index()
        self.data = self.data.rename(index=str, columns={"Timestamp": "ds", "Weighted_Price": "y"})
        self.data["ds"] = pd.to_datetime(self.data["ds"], unit="s")

    def get_current_price(self):
        url = 'https://min-api.cryptocompare.com/data/price?fsym={}&tsyms={}' \
                  .format('BTC'.upper(), ','.join(['USD']).upper()) + '&e={}'.format('Coinbase')
        page = requests.get(url)
        data = page.json()
        return data["USD"]

    def get_previous_price(self, start_date, end_date, period_minutes=1):
        min_time = self.data.get_value(0, "ds").to_pydatetime()
        max_time = self.data.get_value(len(self.data) - 1, "ds").to_pydatetime()
        start_date = dt.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        end_date = dt.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        assert min_time <= start_date
        assert max_time >= end_date
        assert start_date <= end_date
        start_index = self.data.index[self.data["ds"] == start_date][0]
        end_index = self.data.index[self.data["ds"] == end_date][0]
        sliced_data = self.data[start_index:end_index + 1]
        return sliced_data.iloc[::period_minutes]

        '''
    def get_previous_price(period_minutes=1,start_date,end_date):
        
        
        a = exchanges.CoinDesk.get_historical_data_as_dict()
        a = {value: float(a[value]) for value in a}
        previous_prices = pd.DataFrame(list(a.items()), columns=["ds", "y"])
        return previous_prices[previous_prices.index % period == 0]
    '''
        '''
    def minute_price_historical():
        url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}'\
                .format("BTC".upper(), "USD".upper(), 20000, 1)+ '&e={}'.format("Coinbase")


        page = requests.get(url)
        data = page.json()['Data']
        df = pd.DataFrame(data)
        df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
        c = ["timestamp","open","close","high","low","volumefrom","volumeto"]
        df = df.reindex(columns=c)

        return df
'''


