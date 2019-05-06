import sys
sys.path.append('./../')

from dataset.dataset import Dataset
from model.sgd_classifier import Classifier
from evolution.evolution import EvolutionStrategy

def main():
	dataset = Dataset()
	# read hyperparameter from file
	...
	hyperparameter_pool = {}
	evolution_opts = {'offspring_number': 3, 'models_to_keep_number': 2}
	evol = EvolutionStrategy(hyperparameter_pool)
	models = evol.generate_initial_population(num_models=5)

	accuracies = []
	for model in models:
		classifier = Classifier(model)
		classifier.fit(dataset.train_split())
		accuracies.append(classifier.predict(dataset.test_split()))
