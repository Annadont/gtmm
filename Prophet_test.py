from fbprophet import Prophet
import pandas as pd
import numpy as np
data=pd.read_csv("market-price.csv")

data=pd.DataFrame(data, None, ["ds","y"])
data["y"]=np.log(data["y"])
#data["ds"]=pd.to_datetime(data["ds"])
print(data.head())

navi = Prophet()

navi.fit(data)

future = navi.make_future_dataframe(periods=365)
forecast= navi.predict(future)
print(forecast)
navi.plot(forecast).show()
