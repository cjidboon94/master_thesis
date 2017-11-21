import random
import numpy as np

class Individual():
    def __init__(self, genes, timestamp):
        self.genes = genes
        self.timestamp = timestamp
        self.score = None

    def evaluate(self):
        raise NotImplementedError()

    def mutate(self, mutation_size):
        raise NotImplementedError()

def sort_individuals(individuals):
    """
    Sort the individuals

    Parameters:
    ----------
    individuals: a list of Individual Objects

    Returns: a sorted list of Individual objects

    """
    for individual in individuals:
        if individual.score is None:
            raise ValueError("all individuals should have scores")

    return sorted(individuals, key=lambda x: x.score)

def select_parents_universal_stochastic(amount_of_parents, sorted_population,
                                        rank_probabilities):
    """select the selected parents using stochastic universal selection 
    
    Parameters:
    ----------
    amount_of_parents: integer
    sorted_population: a list of tuples with scores and numpy arrays
    rank_probabilities: a numpy array
        The probabilities to assign to every item.

    """
    population_rank_probabilities = zip(sorted_population, rank_probabilities)
    points = (np.linspace(0, 1, amount_of_parents, False) +
              np.random.uniform(0, 1.0/amount_of_parents))

    #print("the shape of the points {}".format(points.shape))

    random.shuffle(population_rank_probabilities)
    population = zip(*population_rank_probabilities)[0]
    rank_probabilities = zip(*population_rank_probabilities)[1]
    bins = np.zeros(len(sorted_population)+1)
    probability_mass = 0 
    for i in range(len(sorted_population)):
        bins[i+1] = rank_probabilities[i] + probability_mass
        probability_mass += rank_probabilities[i]

    parent_indices, _ = np.histogram(points, bins)
    parents = []
    for index, amount_of_samples in enumerate(parent_indices):
        for i in range(amount_of_samples):
            parents.append(population[index])

    return parents

def select_new_population(old_individuals, new_individuals, size, 
                          generational):
    """
    Select the new individuals

    Parameters:
    ----------
    old_individuals: list of Individual Objects
    new_individuals: list of Individual Objects
    size: number
        The population size
    generational: Boolean
        whether to discard the old population

    Returns:
    -------
    a list of Individual objects

    """
    if generational:
        individuals = new_individuals
    else:
        individuals = old_individuals+new_individuals

    return sort_individuals(individuals)[:size]


def select_parents(individuals, number_of_parents, mode):
    """
    selectDefaults to random selection
    
    Parameters:
    ----------
    individuals: list of Individual objects
    number_of_parents: a number
    mode: string
        Either "rank_exponential" or it will be done randomly

    Returns: list of Individual objects

    """
    #print("number of parents {}".format(number_of_parents))
    if mode=="rank_exponential":
        rank_scores_exponential = 1-np.exp(-1*np.arange(len(individuals)))
        scores = rank_scores_exponential/np.sum(rank_scores_exponential)
        #print("length of scores {} ".format(scores.shape))
        #print(scores)
    else:
        scores = np.array([1.0/len(individuals)] * number_of_parents)
        #print("the length of scores {}".format(scores.shape))

    parents = select_parents_universal_stochastic(
        number_of_parents, sort_individuals(individuals), scores
    )
    return parents




