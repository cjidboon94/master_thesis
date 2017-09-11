import nudge_new as nudge
import numpy as np

class OptimizeVector():
    def __init__(initial_population, mutate, evaluate, select_parents,
                 select_new_population):
        """
        Optimize a vector using an evolutionary algorithm

        Parameters:
        ----------
        initial_population: list of 1-d numpy arrays
        mutate: a function 
            to mutate one member of the population (a 1-d numpy array)
        evalutate: a function

        """
        pass

class Individual():
    def __init__(self, genes, timestamp):
        self.genes = genes
        self.timestamp = timestamp
        self.score = None

class Population():
    def __init__(self, size):
        self.population = create_initial_population(size)
    
    def evolve(self, number_of_children, generational):
        parents = select_parents(self.population, number_of_children*2)
        children = create_children(parents, number_of_children)
        evaluate(children)
        self.population = select_new_population(
            self.population, children, len(self.population), generational
        )

    def get_sorted():
        for individual in population:
            if individual.score is None:
                raise ValueError("all individuals should have scores")

        return sorted(self.population, key=lambda x: x["score"])

def create_initial_population(size):
    raise NotImplementedError()

def select_parents(population, number_of_parents, mode):
    """
    Defaults to random selection

    """
    if mode=="rank_exponential":
        rank_scores_exponential = 1-np.exp(-1*np.arange(population_size))
        scores = rank_scores_exponential/np.sum(rank_scores_exponential)
    else:
        scores = [1/len(population)] * number_of_parents

    sorted_population = population.get_sorted()
    parents = select_parents_universal_stochastic(
        number_of_parents, sorted_population, scores
    )
    return parents

def evaluate(individuals):
    """
    add score to individual objects

    Parameters:
    ----------
    individuals: a list of Individual objects

    """
    raise NotImplementedError()

def create_children(individuals, number_of_children):
    """
    Create new individuals.

    Parameters:
    ----------
    individuals: a list of individual objects

    """
    raise NotImplementedError()

def select_new_population(old_population, new_population, size, generational):
    raise NotImplementedError()

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

    random.shuffle(population_rank_probabilities)
    population = zip(*population_rank_probabilities)[0]
    rank_probabilities = zip(*population_rank_probabilities)[1]
    bins = np.zeros(len(sorted_population))
    probability_mass = 0 
    for i in range(len(sorted_population)):
        bins[i] = rank_probabilities[i] + probability_mass
        probability_mass += rank_probabilities[i]

    parent_indices, _ = np.histogram(points, bins)
    parents = []
    for index, amount_of_samples in enumerate(parent_indices):
        for i in range(amount_of_samples):
            parents.append(population[index])

    return parents

def mutate_distribution(distribution, mutation_size):
    """
    Mutate the probability distribution

    Parameters:
    ----------
    distribution: a numpy array
    mutation_size: a float

    """
    mutation = np.random.uniform(-mutation_size, mutation_size, distribution.shape)
    mutation = mutation
    mutated_distribution = np.copy(distribution)

    mutated_distribution = np.minimum(np.maximum(mutated_distribution + mutation, 0), 1)
    mutated_distribution = mutated_distribution/np.sum(mutated_distribution)
    if abs(np.sum(mutated_distribution)-1) > 10**-6:
        raise ValueError()

    return mutated_distribution

def produce_distribution_with_entropy_evolutionary(
        shape, entropy_size, number_of_trials, 
        population_size=10, number_of_children=20,
        generational=False, initial_dist='peaked', number_of_peaks=1
        ):
    """
    Produce a distribution with a given entropy

    Parameters:
    ----------
    shape: a tuple of ints
    entropy_size: the entropy size- base 2
    number_of_trials: integer
    population_size: integer
    number_of_children: integer
    generational: boolean
        Whether to replace every sample in the population

    Returns: 
    -------
    a numpy array representing a probability distribution with
    a certain entropy
    """
    total_number_of_states = reduce(lambda x,y: x*y, shape)
    if initial_dist=='peaked':
        population = []
        for i in range(population_size):
            sample = np.zeros(total_number_of_states)
            peaks = np.random.randint(0, total_number_of_states, number_of_peaks)
            peak_size = 1.0/number_of_peaks
            for peak in peaks:
                sample[peak] = peak_size
            population.append(sample)
    elif initial_dist=='random':
        population = [compute_joint_uniform_random(shape).flatten()
                      for i in range(population_size)]

    rank_scores_exponential = 1-np.exp(-1*np.arange(population_size))
    rank_exp_probabilities = rank_scores_exponential/np.sum(rank_scores_exponential)
    for i in range(number_of_trials):
        population_scores = [abs(entropy_size-entropy(dist.flatten(), base=2)) 
                             for dist in population]
        sorted_population_scores = list(sorted(zip(population_scores, population), 
                                               key=lambda x:x[0]))
        sorted_population = zip(*sorted_population_scores)[1]
        parents = select_parents(number_of_children, sorted_population,
                                 rank_exp_probabilities)
        if i<number_of_trials/3.0:
            mutation_size = .3/total_number_of_states
        elif i<number_of_trials/2.0:
            mutation_size = 0.15/total_number_of_states
        elif i<(number_of_trials*0.75):
            mutation_size = 0.1/total_number_of_states
        else:
            mutation_size = 0.05/total_number_of_states
        children = [mutate_distribution(parent, mutation_size) for parent in parents]
        scores = [abs(entropy_size-entropy(dist.flatten(), base=2)) for dist in children]
        children_scores = list(zip(scores, children))
        if generational:
            new_population_sorted_scores = sorted(children_scores, key=lambda x: x[0])
        else:
            new_population_sorted_scores = sorted(children_scores+sorted_population_scores, key=lambda x: x[0])


        population = zip(*new_population_sorted_scores)[1][:population_size]
        #print(population[0])
        if i%20==0:
            pass
            #print(entropy(population[0], base=2), end=" ")

    return population[0]

if __name__ == "__main__":
    number_of_trials = 100
    population = Population(10)
    for i in range(number_of_trials):
        population.evolve()
