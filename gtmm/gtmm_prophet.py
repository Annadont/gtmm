from fbprophet import Prophet
from pandas import DataFrame
import numpy as np

class GtmmProphet:

    def __init__(self, dataset):
        assert isinstance(dataset, DataFrame)
        self.data = dataset
        self.navi = Prophet()

    def train_prophet(self):
        self.data["y"] = np.log(self.data["y"])
        self.navi.fit(self.data)


    def get_prophet(self):
        return self.navi

