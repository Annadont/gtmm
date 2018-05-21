import exchanges
import pandas as pd


def get_current_price():
    return float(exchanges.CoinDesk.get_current_price())


def get_previous_price(period=13):
    a = exchanges.CoinDesk.get_historical_data_as_dict()
    a = {value: float(a[value]) for value in a}
    previous_prices = pd.DataFrame(list(a.items()), columns=["ds", "y"])
    return previous_prices[previous_prices.index % period == 0]
