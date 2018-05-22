from gtmm.gtmm_svm import GtmmSVM, combineInputSources
from gtmm.public_sentiment import PubSent
from gtmm.current_price import CurrPrice
from gttm.gtmm_prophet import GtmmProphet
from datetime import datetime, timedelta

def main():

  prohpet_prices = CurrPrice().get_previous_price(period=1, end_date=datetime.now())
  prophet_trainer = GtmmProphet(prohpet_prices)
  prophet_trainer.train_prophet()
  prophet_trained = prophet_trainer.get_prophet()

  pub_sent = PubSent() ## figure out how to get data to dawson
  pub_sent_yhat = pub_sent.predict() ## get some data in the target time


  week_prices = CurrPrice().get_previous_price(period=60, start_date=datetime.now() - timedelta(hours=28), end_date=datetime.now())['y']
  curr_time = datetime.now()
  curr_price = CurrPrice().get_current_price(date=datetime.now())

  svm_input_X = combineInputSources(prophet_trained, pub_sent_yhat, week_prices, curr_price, curr_time)

  clf = GtmmSVM()
  clf.train(svm_input_X, ) ## figure out how to calculate y

main()