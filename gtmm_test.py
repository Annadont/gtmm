from gtmm.gtmm_svm import GtmmSVM, combine_input_sources, calculate_y
from gtmm.public_sentiment import PubSent
from gtmm.current_price import CurrPrice
from gttm.gtmm_prophet import GtmmProphet
from datetime import datetime, timedelta
import numpy as np

curr_price_tool = CurrPrice()

def make_x(curr_time):
  prohpet_prices = curr_price_tool.get_previous_price(period=1, end_date=curr_time)
  prophet_trainer = GtmmProphet(prohpet_prices)
  prophet_trainer.train_prophet()
  prophet_trained = prophet_trainer.get_prophet()

  pub_sent = PubSent() ## figure out how to get data to dawson
  previous_time = curr_time - timedelta(hours=6)
  pub_sent_X = pub_sent.makeInputData(previous_time)
  pub_sent_y = pub_sent.calculateY(pub_sent_X, previous_time)
  pub_sent.train(pub_sent_X, pub_sent_y)
  
  pub_sent_yhat = pub_sent.predict(curr_time) ## get some data in the target time

  week_prices = curr_price_tool.get_previous_price(period=60, start_date=curr_time - timedelta(hours=28), end_date=curr_time)['y']
  curr_price = curr_price_tool.get_current_price(date=curr_time)

  return combine_input_sources(prophet_trained, pub_sent_yhat, week_prices, curr_price, curr_time)


def make_x_y(curr_time):
  X = make_x(curr_time)
  curr_price = curr_price_tool.get_current_price(date=curr_time)
  future_price = curr_price_tool.get_current_price(date=curr_time + timedelta(hours=6))
  y = calculate_y(curr_price, future_price)

  return X, y


def predict(clf, curr_time):
  X = make_x(curr_time)
  return clf.predict(X)


def main():

  prohpet_prices = curr_price_tool.get_previous_price(period=1, end_date=datetime.now())
  train_examples_len = len(prohpet_prices)

  X_arr = []
  y_arr = []
  for i in range(0, train_examples_len):
    curr_time = prohpet_prices[i]['ds']
    X, y = make_x_y(curr_time)
    X_arr.append(X)
    y_arr.append(y)

  X = np.array(X_arr)
  y = np.array(y_arr)

  X_train = X[:-20]
  y_train = X[:-20]

  X_test = X[-20:]
  y_test = X[-20:]

  clf = GtmmSVM()
  clf.train(X_train, y_train) ## figure out how to calculate y
  
  predictions = clf.predict(X_test)
  faults = clf.test(predictions, y_test)
  print "Number incorrect: " + str(faults/len(X_test))

main()