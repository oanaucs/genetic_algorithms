from sklearn.datasets import fetch_openml

class Dataset():
	def __init__(self, dataset_name='cifar_10_small'):
		self.data = fetch_openml(name=dataset_name)

	def data(self):
		return self.data

	def shape(self):
		return self.data.data.shape

	def categories(self):
		return self.data.categories

	def feature_names(self):
		return self.data.feature_names

	def target(self):
		return self.data.target

	def description(self):
		return self.data.DESCR 

	def details(self):
		return self.data.details
