from gtmm.gtmm_svm import GtmmSVM, combine_input_sources, calculate_y
# from gtmm.public_sentiment import PubSent
from gtmm.current_price import CurrentPrice
from gtmm.gtmm_prophet import GtmmProphet
from datetime import datetime, timedelta
import numpy as np
import pickle
import os.path

curr_price_tool = CurrentPrice('./data/coinbase.csv')
start_timestamp = 1521111120
end_timestamp = 1521356880

def make_x(curr_time, starttime):
  '''Returns an X input row given a time'''
  prohpet_prices = curr_price_tool.get_previous_price(period_minutes=1, start_date=starttime, end_date=curr_time)
  prophet_trainer = GtmmProphet(prohpet_prices)
  prophet_trainer.train_prophet()
  prophet_trained = prophet_trainer.get_prophet()

  # pub_sent = PubSent() ## figure out how to get data to dawson
  # previous_time = curr_time - timedelta(hours=6)
  # pub_sent_X = pub_sent.makeInputData(previous_time)
  # pub_sent_y = pub_sent.calculateY(pub_sent_X, previous_time)
  # pub_sent.train(pub_sent_X, pub_sent_y)

  pub_sent_yhat = 1 # pub_sent.predict(curr_time) ## get some data in the target time

  week_prices = curr_price_tool.get_previous_price(period_minutes=60, start_date=curr_time - timedelta(hours=28), end_date=curr_time)['y']
  curr_price = curr_price_tool.get_previous_price(period_minutes=60, start_date=curr_time, end_date=curr_time)

  return combine_input_sources(prophet_trained, pub_sent_yhat, week_prices, curr_price, curr_time)


def make_x_y(curr_time, starttime):
  '''Returns an X input and y output row given a time (assuming there is a future time'''
  X = make_x(curr_time, starttime)
  curr_price = curr_price_tool.get_previous_price(period_minutes=60, start_date=curr_time + timedelta(hours=6), end_date=curr_time + timedelta(hours=6))
  future_price = curr_price_tool.get_previous_price(period_minutes=60, start_date=curr_time + timedelta(hours=12), end_date=curr_time + timedelta(hours=12))
  y = calculate_y(curr_price['y'].values, future_price['y'].values)

  return X, y

def predict(clf, curr_time, starttime):
  '''Predicts a value in the future based on a trained svm and the time to predict'''
  X = make_x(curr_time, starttime)
  return clf.predict(X)


def main():
  '''Sets up the inputs, outputs, svm, and data. Trains, tests, and reports accuracy'''

  prohpet_prices = curr_price_tool.get_previous_price(period_minutes=60, start_date=datetime.fromtimestamp(start_timestamp), end_date=datetime.fromtimestamp(end_timestamp))
  train_examples_len = len(prohpet_prices)
  
  X = None
  y = None
  
  if not os.path.isfile('./X_dat.dat') or not os.path.isfile('./y_dat.dat'):
    X_arr = []
    y_arr = []
    for i in range(20, train_examples_len):
      print str(i) + " of " + str(train_examples_len)
      curr_time = prohpet_prices['ds'][i].to_pydatetime()
      X, y = make_x_y(curr_time, datetime.fromtimestamp(end_timestamp))
      X_arr.append(X)
      y_arr.append(y)

    X = np.array(X_arr)
    y = np.array(y_arr)

    X[np.isnan(X)] = 0

    X.dump('X_dat.dat')
    y.dump('y_dat.dat')
  else:
    X = np.load('X_dat.dat')
    y = np.load('y_dat.dat')


  X_train = X[:-20]
  y_train = y[:-20]

  X_test = X[-20:]
  y_test = y[-20:]

  clf = GtmmSVM()
  clf.train(X_train, y_train) ## figure out how to calculate y
  
  predictions = clf.predict(X_test)
  faults = clf.test(predictions, y_test)
  print "Number incorrect: " + str(faults/len(X_test))

main()