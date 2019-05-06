import numpy as np

from itertools import combinations

def EvolutionStrategy():
	def __init__(self, hyperparameter_pool, evolution_opts):
		self.hyperparameter_pool = hyperparameter_pool
		self.hyperparamer_names = list(self.hyperparameter_pool.keys())
		self.evolution_opts = evolution_opts
		self.current_population = None

	def select_random_hyperparameters(self):
		hyperparameters = {}
		for hyperparam_name, possible_values in self.hyperparameter_pool.items():
			hyperparameters.update({hyperparam_name: np.random.choice(possible_values)})

		return hyperparameters

	def generate_initial_population(self, num_models):
		population = []
		for i in range(0, num_models):
			population.append(self.select_random_hyperparameters())
		self.current_population = population
		np.random.shuffle(self.current_population)
		return self.current_population

	def create_offspring(self, models_idxs):
		# create possible combinations of models
		offspring_combinations = list(combinations(models_idxs, 2))
		# select 50/50 from each of the offspring combination models
		offspring = []
		for i in range(0, self.evolution_opts['offspring_number']):
			comb = np.random.choice(offspring_combinations, 1)
			# randomly choose which keys will be taken from first model
			selection = np.random.choice(self.hyperparamer_names, int(len(self.hyperparamer_names)/2))
			current_offspring = dict()
			for s in selection:
				current_offspring.update({s: self.current_population[comb[0]][s]})
			for s in self.hyperparamer_names:
				if s not in selection:
					current_offspring.update({s: self.current_population[comb[1][s]]})
		return offspring

	def generate_population(self, current_performances):
		models_idxs = np.argsort(current_performances)[:5]
		offspring = create_offspring(models_idxs)

		models_to_keep = np.random.choice(self.current_population, self.evolution_opts['models_to_keep_number']) 
		self.current_population = models_to_keep.expand(offspring)
		np.random.shuffle(self.current_population)

		return self.current_population

