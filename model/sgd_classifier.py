from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


def Classifier():
	def __init__(self, hyperparamters = None):
		self.classifier = SGDClassifier(hyperparameters)

	def fit(self, samples, labels):
		info = self.classifier.fit(samples, labels)
		print(info)

	def predict(self, sample):
		return self.classifier.predict(sample)

	def evaluate_accuracy(self, samples, labels):
		predictions = []
		for sample in samples:
			predictions.append(self.predict(sample))
		return accuracy_score(labels, predictions)
