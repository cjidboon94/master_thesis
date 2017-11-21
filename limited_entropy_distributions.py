"""
This module's purpose is to create probability distributions with a certain
entropy using evolutionary algorithms

"""
import numpy as np
from scipy import stats
import evolutionary_algorithms as ea

def get_dist_percentage_max_entropy(dist_shape, percentage_max_entropy,
                                    evolutionary_parameters, verbose=False):
    number_of_states = reduce(lambda x,y: x*y, dist_shape)
    goal_entropy = np.log2(number_of_states)*percentage_max_entropy
    dist = get_dist_with_entropy(number_of_states, goal_entropy, 
                                 evolutionary_parameters, verbose)
    return np.reshape(dist, dist_shape)

def get_dist_with_entropy(number_of_states, entropy_size, 
                          evolution_params, verbose=False):
    """
    Create a distribution (randomly) with a certain entropy

    Parameters:
    ----------
    number_of_states: int
    entropy_size: a postive float

    Returns: 1d nd-array, respresenting a probability distribution

    """
    individuals = [Distribution.create_random_distribution(number_of_states)
                   for _ in range(evolution_params["population_size"])]

    for individual in individuals:
        individual.evaluate(entropy_size)

    population = DistributionPopulation(individuals)
    best_initial_dist = population.get_best_distribution()
    if verbose:
        print("best initial score: {}".format(best_initial_dist.score))

    for _ in range(evolution_params["number_of_generations"]):
        population.evolve(
            entropy_size, evolution_params["number_of_children"],
            evolution_params["parent_selection_mode"],
            evolution_params["population_size"],
            evolution_params["generational"],
            evolution_params["number_of_mutations"],
            evolution_params["mutation_size"]
        )
        if verbose:
            best_score=population.get_best_distribution().score
            print("best score: {}".format(best_score))
            
    return population.get_best_distribution().distribution

def mutate_distribution_uniform_entropy(distribution, mutation_size):
    """
    Mutate the probability distribution

    Parameters:
    ----------
    distribution: a 1d nd-array
    mutation_size: a float

    """
    mutation = np.random.uniform(-mutation_size, mutation_size, distribution.shape)
    mutation = np.minimum(mutation, distribution)
    mutated_distribution = np.copy(distribution)

    mutated_distribution = np.minimum(np.maximum(mutated_distribution + mutation, 0), 1)
    mutated_distribution = mutated_distribution/np.sum(mutated_distribution)
    if abs(np.sum(mutated_distribution)-1) > 10**-7:
        raise ValueError()
    if np.any(mutated_distribution < 0):
        raise ValueError()

    return mutated_distribution

class Distribution(ea.Individual):
    def __init__(self, distribution):
        """create an individual for an individual algorithms

        Parameters:
        ----------
        distribution: 1d nd-array, representing a probability distribution

        """
        self.distribution = distribution
        self.score = None

    @classmethod
    def create_random_distribution(cls, number_of_states):
        new_dist = np.random.dirichlet(number_of_states*[1])
        return cls(new_dist)

    def mutate_step_wise(self, number_of_mutations, max_mutation_size):
        negative_states = np.random.choice(
            self.distribution.nonzero()[0], number_of_mutations, replace=False
        )
        positive_states = np.random.choice(
            self.distribution.shape[0], number_of_mutations, replace=False
        )
        mutations = np.random.uniform(0, max_mutation_size, 
                                      number_of_mutations)
        mutations = np.minimum(mutations, self.distribution[negative_states])
        self.distribution[negative_states] = (
            self.distribution[negative_states] - mutations
        )
        self.distribution[positive_states] = (
            self.distribution[positive_states] + mutations
        )
        if abs(np.sum(self.distribution)-1) > 10**-7:
            raise ValueError()
        if np.any(self.distribution < 0):
            raise ValueError()

    def mutate_uniform(self, mutation_size):
        """
        Mutate the probability distribution

        Parameters:
        ----------
        mutation_size: a float

        """
        mutation = np.random.uniform(-mutation_size, mutation_size, 
                                     self.distribution.shape)
        self.distribution = np.minimum(
            np.maximum(self.distribution+mutation, 0), 1
        )
        self.distribution = self.distribution/np.sum(self.distribution)
        if abs(np.sum(self.distribution)-1) > 10**-7:
            raise ValueError()

    def evaluate(self, entropy_goal):
        """calculate the absolute difference with entropy and entropy goal"""
        self.score = abs(
            entropy_goal-stats.entropy(self.distribution.flatten(), base=2)
        )

class DistributionPopulation():
    """ 
    A class to represent a list of Distribution objects

    Attributes:
    ----------
    individuals: list of Distribution objects

    """
    def __init__(self, distributions):
        """Create a DistributionPopulation object"""
        self.distributions = distributions

    def evolve(self, entropy_goal, number_of_children, parent_selection_mode,
               new_population_size, generational, number_of_mutations,
               mutation_size):
        """
        Evolve the population
        
        Parameters:
        ----------
        number_of_children: int
        parent_selection_mode: string, "rank_exponential" or None
        new_population_size: int
        mutation_size: a number
        generational: Boolean
            Whether to discard the old individuals at every new timestep

        """
        parents = ea.select_parents(
            self.distributions, number_of_children, parent_selection_mode
        )
        children = self.create_children(parents, number_of_children,
                                        number_of_mutations, mutation_size)
        for child in children:
            child.evaluate(entropy_goal)

        self.distributions = ea.select_new_population(
            self.distributions, children, new_population_size, generational
        )

    def create_children(self, parents, number_of_children, 
                        number_of_mutations, mutation_size):
        """
        Create new individuals

        Parameters:
        ----------
        parents: a list of individual objects
        number_of_children: an integer
        mutation_size: a number

        """
        children = []
        for parent in parents:
            child = Distribution(np.copy(parent.distribution))
            child.mutate_step_wise(number_of_mutations, mutation_size)
            children.append(child)

        return children

    def get_best_distribution(self):
        return min(self.distributions, key=lambda x: x.score)

if __name__ == "__main__":
    evolutionary_parameters = {    
        "number_of_generations": 20,
        "population_size": 20,
        "number_of_children": 80,
        "generational": False,
        "number_of_mutations": 100,
        "mutation_size": 0.005,
        "parent_selection_mode": "rank_exponential",
    }
#    number_of_states = 1000
#    entropy_size = 8.0
#    dist = get_dist_with_entropy(
#        number_of_states, entropy_size, evolutionary_parameters,
#        verbose=True    
#    )

    dist_shape = [3]*6
    percentage_max_entropy = 0.90
    dist = get_dist_percentage_max_entropy(
        dist_shape, percentage_max_entropy, evolutionary_parameters, 
        verbose=True
    )
    print("the found entropy {}".format(stats.entropy(dist.flatten(), base=2)))
    print("the distribution")
    print(dist)


