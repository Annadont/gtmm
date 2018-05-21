import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
import pickle
import os.path


m = None
if os.path.exists('fit_prophet_coinbase.dat'):
  model = file('fit_prophet_coinbase.dat', 'r')
  m = pickle.load(model)

if m is None:
  orig_df = pd.read_csv('./data/coinbase.csv')
  clean_df = pd.DataFrame({
    "ds": pd.to_datetime(orig_df['Timestamp'],unit='s')[10000:][::60],
    'y': orig_df['Close'][10000:][::60]
  })

  clean_df.head()

  m = Prophet()
  print "\nFitting..."
  m.fit(clean_df)
  print "Fit\n"
  f = file('fit_prophet_coinbase.dat', 'w')
  pickle.dump(m, f)

future = m.make_future_dataframe(periods=10, freq='H', include_history=False)
print future
print "\nPredicting..."
forecast = m.predict(future)
print "Predicted\n"
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = m.plot(forecast);
plt.show()

