import random
import nudge_new as nudge
import numpy as np
from scipy.stats import entropy

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
    if abs(np.sum(mutated_distribution)-1) > 10**-7:
        raise ValueError()

    return mutated_distribution


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

def evaluate_cond_distribution(cond_output, number_of_input_dists, goal_distance):
    """

    Parameters:
    ----------
    cond_output: nd-array, 
        a conditional probability distribution (last axis are the conditional
        probabilities)
    """
    input_shape = cond_output.shape[:-1]
    input_dists = []
    for i in range(number_of_input_dists):
        input_dist = np.random.dirichlet(reduce(input_shape, lambda x,y: x*y)*[1])
        input_dists.append(np.reshape(input_dist, input_shape))

    number_of_variables = len(input_dists[0].shape)
    outputs = []
    for input_dist in input_dists:
        joint = probability_distributions.compute_joint(
            input_dist, cond_output, set(list(range()))
        )
        output = probability_distributions.ProbabilityArray(joint).marginalize(
            set([number_of_variables])
        )
        outputs.append(output)

    average_dist = np.mean(np.array([a1, a2, a3]), axis=0)
    distance = np.sum(
        [np.sum(np.absolute(output-average_dist)) for output in outputs]
    )
    return abs(distance-goal_distance)

class ConditionalOutput():
    def __init__(self, cond_output):
        """create an individual for an individual algorithms

        Parameters:
        ----------
        genes: a 1d nd-array
        timestamp: a number

        """
        self.cond_output = cond_output
        self.score = None

    def mutate(self, mutation_size):
        self.genes = mutate_distribution_uniform(self.genes, mutation_size)

    def evaluate(self, number_of_input_dists, goal_distance):
        """calculate the absolute difference with entropy and entropy goal"""
        self.score = evaluate_cond_distribution(
            self.cond_output, number_of_input_dists, goal_distance
        )

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

def create_individual_distributions(
            number_of_individuals, number_of_states, 
            number_of_input_variables
            ):
    """
    Create a list of individual objects. The genes are either generated 
    according to the mode.

    Parameters:
    ----------
    number_of_individuals: an integer
    number_of_genes: an integer
    mode: a string
        Either "random" or "peaked"
    number_of_peaks: an integer
    timestamp: an integer

    Returns:
    -------
    a list of IndividualDistribution objects

    """
    individuals = []
    for i in range(number_of_individuals):
        cond_shape = [number_of_states]*(number_of_input_variables+1)
        cond_output = [
            probability_distributions.compute_joint_uniform_random((number_of_states,))
            for i in range(number_of_states**(number_of_input_variables))
        ]
        cond_output = np.array(cond_output)
        cond_output = np.reshape(cond_output, cond_shape)        
        individual = IndividualDistribution(cond_output)
        individuals.append(individual)

    return individuals

class FindDistribution():
    """ 
    A class that uses evolutionary algorithms to find a distribution 
    with a certain entropy

    Attributes:
    ----------
    individuals: list of IndividualDistribution objects
    goal_entropy: a number
    number_of_generations: an integer
    number_of_children: an integer
    parent_selection_mode: Either "rank_exponential" or None

    """
    def __init__(self, individuals, goal_entropy, number_of_generations,
                 number_of_children, parent_selection_mode):
        """Create a FindDistributionEntropy object"""
        self.individuals = individuals
        self.entropy_goal = goal_entropy
        self.number_of_generations = number_of_generations
        self.number_of_children = number_of_children
        self.parent_selection_mode = parent_selection_mode

    def evolve(self, generational, mutation_size):
        """
        Evolve the population
        
        Parameters:
        ----------
        mutation_size: a number
        generational: Boolean
            Whether to discard the old individuals at every new timestep

        """
        for timestep in range(self.number_of_generations):
            parents = select_parents(
                self.individuals, self.number_of_children*2,
                self.parent_selection_mode
            )
            children = self.create_children(parents, self.number_of_children,
                                            timestep, mutation_size)
            for child in children:
                child.evaluate(self.entropy_goal)

            self.individuals = select_new_population(
                self.individuals, children, len(self.individuals), 
                generational
            )

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
        
        return IndividualDistribution(genes, timestamp)

def produce_distribution_with_entropy(shape, percentage_max_entropy):
    """
    Create a distribution (randomly) with a certain entropy

    Parameters:
    ----------
    shape: an iterable
    percentage_max_entropy: a number
        The percentage of the maximum entropy

    Note: for percentage max entropy above 0.9 it does not work well

    """
    distribution_shape = shape
    number_of_generations = 1800
    population_size = 10
    number_of_children = 20
    generational = False
    mutation_size = 0.0020
    if percentage_max_entropy>=0.9:
        mutation_size = 0.10
    parent_selection_mode = "rank_exponential"
    number_of_states = reduce(lambda x,y: x*y, distribution_shape)
    goal_entropy = np.log2(number_of_states)*percentage_max_entropy
    #print("the goal entropy {}".format(goal_entropy))
    individuals = create_individual_distributions(
        population_size, number_of_states, "random"
    )
    for individual in individuals:
        individual.evaluate(goal_entropy)

    #print("entropy initial population {}".format(
    #    entropy(sort_individuals(individuals)[0].genes, base=2)
    #))

    evolve = FindDistributionEntropy(
        individuals, goal_entropy, number_of_generations,
        number_of_children, parent_selection_mode
    )
    evolve.evolve(generational, mutation_size)
    return evolve.individuals[0].genes

if __name__ == "__main__":
    distribution_shape = [5]*6
    number_of_generations = 2000
    percentage_max_entropy = 0.55
    population_size = 10
    number_of_children = 20
    generational = False
    mutation_size = 0.0025
    if percentage_max_entropy>=0.9:
        mutation_size = 0.10
    parent_selection_mode = "rank_exponential"
    number_of_states = reduce(lambda x,y: x*y, distribution_shape)
    goal_entropy = np.log2(number_of_states)*percentage_max_entropy
    print("the goal entropy {}".format(goal_entropy))
    individuals = create_individual_distributions(
        population_size, number_of_states, "random"
    )
    for individual in individuals:
        individual.evaluate(goal_entropy)

    print("entropy initial population {}".format(
        entropy(sort_individuals(individuals)[0].genes, base=2)
    ))

    evolve = FindDistributionEntropy(
        individuals, goal_entropy, number_of_generations,
        number_of_children, parent_selection_mode
    )
    evolve.evolve(generational, mutation_size)
    print(evolve.individuals[0].genes[:1000])
    print("the final entropy {}".format(
        entropy(evolve.individuals[0].genes, base=2)
    ))

    genes = produce_distribution_with_entropy(
        distribution_shape, percentage_max_entropy
    )
    print("the final entropy {}".format(entropy(genes, base=2)))
