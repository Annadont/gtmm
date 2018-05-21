from sklearn import svm

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

