import random
import nudge_new as nudge
import numpy as np
from scipy.stats import entropy
from probability_distributions import compute_joint_uniform_random
from probability_distributions import select_parents

class Individual():
    def __init__(self, genes, timestamp):
        self.genes = genes
        self.timestamp = timestamp
        self.score = None

    def evaluate(self):
        raise NotImplementedError()

    def mutate(self, mutation_size):
        raise NotImplementedError()

class Population():
    def __init__(self, individuals):
        """
        Population to find optimal solution

        Parameters:
        ----------
        individuals: a list of individuals 

        """
        self.individuals = individuals
    
    def get_sorted(self):
        for individual in self.individuals:
            if individual.score is None:
                raise ValueError("all individuals should have scores")

        return sorted(self.individuals, key=lambda x: x.score)

class EvolvePopulation():
    def __init__():
        pass

    def create_children(individuals, number_of_children, timestamp):
        raise NotImplementedError()

    def select_new_population(self, old_population, 
                              new_population, size, generational):
        raise NotImplementedError()

    def recombine(self, parent1, parent2, timestamp):
        raise NotImplementedError()

    def evolve(self, number_of_children, generational):
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

def mutate_distribution_uniform(distribution, mutation_size):
    """
    Mutate the probability distribution

    Parameters:
    ----------
    distribution: a 1d nd-array
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

class IndividualDistributionEntropy(Individual):
    def __init__(self, genes, timestamp):
        """create an individual for an individual algorithms

        Parameters:
        ----------
        genes: a 1d nd-array
        entropy_goal: a number
        timestamp: a number

        """
        self.genes = genes
        self.timestamp = timestamp
        self.score = None

    @classmethod
    def create_random_individual(cls, length_of_genes, mode,
                                 number_of_peaks, timestamp):
        genes = cls.create_genes(length_of_genes, mode, number_of_peaks)
        return cls(genes, timestamp)

    def mutate(self, mutation_size):
        self.genes = mutate_distribution_uniform(self.genes, mutation_size)
        return self

    def evaluate(self, entropy_goal):
        """calculate the absolute difference with entropy and entropy goal"""
        self.score = abs(entropy_goal-entropy(self.genes.flatten(), base=2))

    @staticmethod
    def create_genes(length_of_genes, mode="random", number_of_peaks=None):
        if mode=='peaked':
            genes = np.zeros(length_of_genes)
            peaks = np.random.randint(0, length_of_genes, number_of_peaks)
            peak_size = 1.0/number_of_peaks
            for peak in peaks:
                genes[peak] = peak_size
        elif mode=='random':
            return np.random.dirichlet(length_of_genes*[1])

class PopulationEntropy(Population):
    def __init__(self, individuals):
        """
        Population to find optimal solution

        Parameters:
        ----------
        population: a list of IndividualDistributionEntropy objects

        """
        self.individuals = individuals

    @classmethod
    def create_random_population(cls, population_size, length_of_genes, mode,
                                 number_of_peaks=1, timestamp=0):
        """
        Population to find optimal solution

        Parameters:
        ----------
        population_size: an integer

        """
        population = []
        for i in range(population_size):
            individual = IndividualDistributionEntropy.create_random_individual(
                length_of_genes, mode, number_of_peaks, timestamp
            )
            population.append(individual)

        return cls(population)

    def __len__(self):
        return len(self.individuals)

class EvolvePopulationEntropy(EvolvePopulation):
    def __init__(self, population, goal_entropy, number_of_generations,
                 number_of_children, parent_selection_mode):
        """
        Evolve the population to bring the resulting individuals as
        close to the goal entropy as possible

        Parameters:
        ----------
        population: A PopulationEntropy object
        goal_entropy: a number
        number_of_generations: an integer
        number_of_children: an integer

        """
        self.population = population
        self.entropy_goal = goal_entropy
        self.number_of_generations = number_of_generations
        self.number_of_children = number_of_children
        self.parent_selection_mode = parent_selection_mode

    def evolve(self, generational, mutation_size):
        """
        Evolve the population
        
        Parameters:
        ----------
        number_of_children: an integer
        generational: Boolean
            Whether to discard the old population at every new timestep
        parent_selection_mode: Either "rank_exponential" or None

        """
        for timestep in range(self.number_of_generations):
            parents = self.select_parents(
                self.population.individuals, number_of_children*2,
                self.parent_selection_mode
            )
            children = self.create_children(parents, self.number_of_children,
                                            timestep, mutation_size)
            for child in children:
                child.evaluate(self.entropy_goal)

            self.population = self.select_new_population(
                self.population.individuals, children, len(self.population),
                generational
            )

    def select_new_population(self, old_individuals,
                              new_individuals, size, generational):
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

        """
        if generational:
            individuals = new_individuals
        else:
            individuals = old_individuals+new_individuals

        sorted_population = Population(individuals).get_sorted()
        return PopulationEntropy(sorted_population[:size])

    def create_children(self, parents, number_of_children, timestamp,
                        mutation_size):
        """
        Create new individuals.

        Parameters:
        ----------
        parents: a list of individual objects
        number_of_children: an integer
        timestamp: a number
        mutation_size: a number

        """
        children = []
        for i in range(number_of_children):
            children.append(self.recombine(
                parents[i], parents[number_of_children+i], timestamp
            ))

        children = [child.mutate(mutation_size) for child in children]
        return children

    def recombine(self, parent1, parent2, timestamp):
        """recombine two individuals to create a new individual
        
        Parameters:
        ----------
        parent1: Individual object
        parent2: Individual object
        timestamp: a number

        Returns: Individual object

        """
        if np.random.random() > 0.5:
            genes = np.copy(parent1.genes)
        else:
            genes =  np.copy(parent2.genes)
        
        return IndividualDistributionEntropy(
            parent1.genes, timestamp
        )

    def select_parents(self, individuals, number_of_parents, mode):
        """
        Defaults to random selection
        
        Parameters:
        ----------
        population: list of Individual objects
        number_of_parents: a number
        mode: string
            Either "rank_exponential" or it will be done randomly

        Returns: list of Individual objects

        """
        #print("number of parents {}".format(number_of_parents))
        if mode=="rank_exponential":
            rank_scores_exponential = 1-np.exp(-1*np.arange(len(individuals)))
            scores = rank_scores_exponential/np.sum(rank_scores_exponential)
        else:
            scores = [1/len(individuals)] * number_of_parents

        sorted_individuals = Population(individuals).get_sorted()
        #print("length sorted individuals {}".format(len(sorted_individuals)))
        parents = select_parents_universal_stochastic(
            number_of_parents, sorted_individuals, scores
        )
        return parents

def produce_distribution_with_entropy_evolutionary(
        shape, entropy_size, number_of_trials, 
        population_size=10, number_of_children=20,
        generational=False, initial_dist='random', number_of_peaks=1
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
        children = [
            mutate_distribution_uniform(parent, mutation_size) 
            for parent in parents
        ]
        scores = [abs(entropy_size-entropy(dist.flatten(), base=2)) for dist in children]
        children_scores = list(zip(scores, children))
        if generational:
            new_population_sorted_scores = sorted(children_scores, key=lambda x: x[0])
        else:
            new_population_sorted_scores = sorted(
                children_scores+sorted_population_scores, key=lambda x: x[0]
            )

        population = zip(*new_population_sorted_scores)[1][:population_size]
        #print(population[0])
        if i%20==0:
            pass
            #print(entropy(population[0], base=2), end=" ")

    return population[0]

if __name__ == "__main__":
    distribution_shape = [5]*4
    number_of_generations = 1000
    goal_entropy_percentage = 0.85
    population_size = 10
    number_of_children = 20
    generational = False
    mutation_size = 0.05
    parent_selection_mode = "rank_exponential"
    number_of_states = reduce(lambda x,y: x*y, distribution_shape)
    goal_entropy = np.log2(number_of_states)*goal_entropy_percentage
    print("the goal entropy {}".format(goal_entropy))
    initial_population = PopulationEntropy.create_random_population(
        population_size, number_of_states, "random"
    )
    #print(initial_population)
    for individual in initial_population.individuals:
        individual.evaluate(goal_entropy)

    evolve = EvolvePopulationEntropy(
        initial_population, goal_entropy, number_of_generations,
        number_of_children, parent_selection_mode
    )
    evolve.evolve(generational, mutation_size)
    #print(evolve.population.individuals[0].genes)
    print("the final entropy {}".format(
        entropy(evolve.population.individuals[0].genes, base=2
    )))

    dist = produce_distribution_with_entropy_evolutionary(
        distribution_shape, goal_entropy, number_of_generations
    )
    print(entropy(dist, base=2))
