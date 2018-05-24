import numpy as np
from sklearn import svm
from fbprophet import Prophet
import pandas as pd
import datetime

def combine_input_sources(prophet_trained, twitter_setiment, price_every_x_hours, curr_price, curr_time, delta_hours=6):
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


def calculate_y(current_price, future_price):
	'''calculates up down or not based on a the current price and a future price'''
	percent_increase = (future_price - current_price)/current_price
	if (percent_increase > 0.5):
		return 1
	elif (percent_increase < 0.5):
		return -1
	else: 
		return 0


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


	def test(self, predictions, y):
		faults = 0
		for i in range(0, len(predictions)):
			if predictions[i] != y:
				faults += 1
		
		return faults

