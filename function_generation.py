import random
import itertools
import numpy as np
from scipy.stats import entropy
import probability_distributions
import nudge_non_causal as nudge

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

def mutate_conditional_distribution_uniform(distribution, mutation_size):
    """
    Mutate the probability distribution

    Parameters:
    ----------
    distribution: nd-array
        Representing a conditional output diistribtution (with the the
        output on the last axis)
    mutation_size: a positive float

    Note: It happens in place

    """
    input_shape = distribution.shape[:-1]
    number_of_output_states = distribution.shape[-1]
    lists_of_possible_states_per_variable = [
        range(states) for states in input_shape
    ]
    for state in itertools.product(*lists_of_possible_states_per_variable):
        mutation = nudge.find_noise_vector(number_of_output_states, 
                                           mutation_size)
        distribution[state] = nudge.nudge_states(mutation, distribution[state])

        if abs(np.sum(distribution[state])-1) > 10**-7:
            raise ValueError()

    return distribution


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

    return sort_individuals(individuals)[-size:]


class ConditionalOutput():
    """
    Attributes:
    ----------
    cond_output: nd-array, 
        a conditional probability distribution (last axis are the conditional
        probabilities)
    score: a number

    """
    def __init__(self, cond_output):
        """create an individual for an individual algorithms

        Parameters:
        ----------
        cond_output: nd-array
            representing a conditional probability distribution

        """
        self.cond_output = cond_output
        self.score = None

    def mutate(self, mutation_size):
        """mutate the distribution

        Parameters:
        ----------
        mutation_size: a positive float

        """
        self.cond_output = mutate_conditional_distribution_uniform(
            self.cond_output, mutation_size
        )

    def evaluate(self, goal_distance, input_dists=None, 
                 number_of_input_dists=None):
        """

        Parameters:
        ----------
        goal_distance:
        input_dists:
        number_of_input_dists:

        """
        input_shape = list(self.cond_output.shape[:-1])
        if input_dists is None:
            input_dists = []
            for i in range(number_of_input_dists):
                input_dist = np.random.dirichlet(reduce(lambda x,y: x*y, input_shape)*[1])
                input_dists.append(np.reshape(input_dist, input_shape))

        number_of_input_vars = len(input_dists[0].shape)
        outputs = []
        for input_dist in input_dists:
            joint = probability_distributions.compute_joint(
                input_dist, self.cond_output, set(range(0, number_of_input_vars, 1))
            )
            output = probability_distributions.ProbabilityArray(joint).marginalize(
                set([number_of_input_vars])
            )
            outputs.append(output)

        average_dist = np.mean(np.array(outputs), axis=0)
        distance = np.mean(
            [np.sum(np.absolute(output-average_dist)) for output in outputs]
        )
        self.score = abs(distance-goal_distance)

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

def create_condional_distributions(
            number_of_conditional_outputs, number_of_states, 
            number_of_input_variables
            ):
    """
    Create a list of ContionalOutput objects

    Parameters:
    ----------
    number_of_conditional_outputs: an integer
    number_of_states: an integer
    number_of_input_variables: an integer

    Returns:
    -------
    a list of ConditionalOutput objects

    """
    conditional_outputs = []
    for _ in range(number_of_conditional_outputs):
        cond_shape = [number_of_states]*(number_of_input_variables+1)
        cond_output = [
            probability_distributions.compute_joint_uniform_random((number_of_states,))
            for i in range(number_of_states**(number_of_input_variables))
        ]
        cond_output = np.reshape(np.array(cond_output), cond_shape)
        conditional_outputs.append(ConditionalOutput(cond_output))

    return conditional_outputs

class FindConditionalOutput():
    """ 
    A class that uses evolutionary algorithms to find a distribution 
    with a certain entropy

    Attributes:
    ----------
    individuals: list of IndividualDistribution objects
    goal_distance: a number
    number_of_generations: an integer
    number_of_children: an integer
    parent_selection_mode: Either "rank_exponential" or None

    """
    def __init__(self, individuals, goal_distance, number_of_generations,
                 number_of_children, parent_selection_mode):
        """Create a FindConditionalOutput object"""
        self.individuals = individuals
        self.goal_distance = goal_distance
        self.number_of_generations = number_of_generations
        self.number_of_children = number_of_children
        self.parent_selection_mode = parent_selection_mode

    def evolve(self, generational, mutation_size, input_dists,
               number_of_input_dists):
        """
        Evolve the population
        
        Parameters:
        ----------
        mutation_size: a number
        generational: Boolean
            Whether to discard the old individuals at every new timestep
        number_of_input_distributions: an integer

        """
        for timestep in range(self.number_of_generations):
            print("timestep {}, worst {}, best {}".format(
                timestep, self.individuals[0].score, self.individuals[-1].score
            ))
            parents = select_parents(
                self.individuals, self.number_of_children*2,
                self.parent_selection_mode
            )
            children = self.create_children(parents, self.number_of_children,
                                            timestep, mutation_size)
            for child in children:
                child.evaluate(self.goal_distance, input_dists,
                               number_of_input_dists)

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
                parents[i], parents[number_of_children+i]
            ))
        for child in children:
            child.mutate(mutation_size)

        return children

    def recombine(self, parent1, parent2):
        """recombine two individuals to create a new individual
        
        Parameters:
        ----------
        parent1: Individual object
        parent2: Individual object
        timestamp: a number

        Returns: Individual object

        """
        if np.random.random() > 0.5:
            genes = np.copy(parent1.cond_output)
        else:
            genes = np.copy(parent2.cond_output)
        
        return ConditionalOutput(genes)

if __name__ == "__main__":
    pass
