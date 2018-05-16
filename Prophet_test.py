from fbprophet import Prophet
import pandas as pd
import numpy as np
data=pd.read_csv("manning_data.csv")
data["y"]=np.log(data["y"])

navi = Prophet()

navi.fit(data)

future = navi.make_future_dataframe(periods=365)
#print(future.tail())
print(navi.predict(future))