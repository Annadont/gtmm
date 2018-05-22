import numpy as np
from sklearn import svm
from fbprophet import Prophet
import pandas as pd
import datetime

def combineInputSources(prophet_trained, twitter_setiment, price_every_x_hours, curr_price, curr_time, delta_hours=6):
	time_to_predict = datetime.datetime(curr_time) + datetime.timedelta(hours=delta_hours)
	future_df = pd.DataFrame({'ds':[ time_to_predict ]}).tail()
	predicted_df = prophet_trained.predict(future_df)
	prediction = predicted_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()['yhat']

	return np.concatenate([ 
					prediction.values[0], 
					np.array(twitter_setiment), 
		  		np.array(curr_price), 
					price_every_x_hours
				]).T


class GtmmSVM():
	def __init__(self, kernel='linear'): 
		self.clf = svm.SVC(kernel=kernel)
		return
	
	def train(self, X, y):
		self.X = X
		self.y = y
		return self.clf.fit(X, y)


	def predict(self, X):
		self.predict_X = X
		return self.clf.predict(X)


	def get_svc(self):
		return self.clf 

