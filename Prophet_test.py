from fbprophet import Prophet
import pandas as pd
import numpy as np
data=pd.read_csv("bitcoinData.csv")
print(data.head())
data=pd.DataFrame(data,None,["ds","y"])
data["y"]=np.log(data["y"])
navi = Prophet()

navi.fit(data)

#future = navi.make_future_dataframe(periods=365)
#print(future.tail())
#forecast= navi.predict(future)
#navi.plot(forecast)
