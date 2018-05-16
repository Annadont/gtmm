import pandas as pd
import numpy as np
from fbprophet import Prophet

orig_df = pd.read_csv('./data/coinbase.csv')
clean_df = pd.DataFrame({
  "DS": orig_df['Timestamp'][0:500],
  'y': np.log(orig_df['Close'][0:500])
})

clean_df.head()

m = Prophet()
m.fit(clean_df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m.plot(forecast);

