from __future__ import print_function
import random
import numpy as np
from scipy.stats import entropy
from jointpdf.jointpdf import JointProbabilityMatrix
from jointpdf.jointpdf import FullNestedArrayOfProbabilities

 
class ProbabilityArray():
    """
    Represents a discrete joint probability distribution as a tree where every
    level in the tree is one variable and every path to a leaf is a state and
    the value of the leaf is the probability of that state. This is 
    implemented as a numpy array where every axis is a level in the tree 
    (a variable)."""
    def __init__(self, probability_distribution):
        """
        Parameters:
        ----------
        probability_distribution: a numpy array
            for all values x it should hold 0<=x<=1 and all values should
            sum to 1

        """
        if np.absolute(np.sum(probability_distribution)-1) > 10**-9:
            raise ValueError("probability distribution sums to {}".format(
                np.sum(probability_distribution)
            ))
        if np.any(probability_distribution < 0):
            raise ValueError("some probability is smaller than 0") 
        self.probability_distribution = probability_distribution

    def marginalize(self, variables, distribution=None):
        """
        Find the marginal distribution of the variables

        Parameters:
        ----------
        variables: a set of integers
            Every variable in variables should be smaller than the total 
            number of variables in the probability distribution

        Returns:
        -------
        A numpy array with as many axis as there were variables the
        variables have the same order as in the joint

        """
        if distribution is None:
            probability_distribution = self.probability_distribution
        else:
            probability_distribution = distribution

        marginal_distribution = np.zeros(tuple(
            [probability_distribution.shape[i] for i in variables]
        ))
        it = np.nditer(probability_distribution, flags=['multi_index'])
        while not it.finished:
            marginal_state = tuple([it.multi_index[i] for i in variables]) 
            marginal_distribution[marginal_state] += it.value
            it.iternext()

        return marginal_distribution

    def find_conditional(self, marginal_variables, conditional_variables):
	"""create the conditional distribution for the selected_indices given 
	the conditional_indices for the joint_distribution
	
	Parameters:
	----------
	marginal_indices: set of integers
            variables that are not conditioned on but are included
	conditional_indices: set of integers
            variables that are conditioned on
	
	Returns: conditional_distribution, marginal_labels, conditional_labels 
	-------
        conditional_distribution: a numpy array
        marginal_labels: list of integers
            The variables that are NOT conditioned on
        conditional_labels: list of integers
            The variables that are conditioned on

	"""
        joint_distribution, marginal_labels, conditional_labels = (
            self.find_joint_marginal(marginal_variables, conditional_variables)
        ) 
        
	marginal_conditional = self.marginalize(conditional_labels,
                                                joint_distribution)
	conditional_distribution = np.copy(joint_distribution) 
	it = np.nditer(joint_distribution, flags=['multi_index'])
	while not it.finished:
            if it.value == 0:
                it.iternext()
                continue
            conditional_arguments = tuple(
                [it.multi_index[i] for i in conditional_labels]
            )
	    conditional_distribution[it.multi_index] = (
                it.value/marginal_conditional[conditional_arguments]
	    )
	    it.iternext()
	
        conditional_shape = [i for count, i in enumerate(joint_distribution.shape)
                             if count in conditional_labels]
        total_sum_conditional_distribution = reduce(lambda x,y: x*y, 
                                                    conditional_shape)
        if abs(np.sum(conditional_distribution)-total_sum_conditional_distribution)> 10**(-8):
            raise ValueError("sum is {} while it should be {}".format(
                np.sum(conditional_distribution), total_sum_conditional_distribution
            ))

    	return (conditional_distribution, marginal_labels, conditional_labels)

    def find_joint_marginal(self, variables1, variables2, distribution=None):
        """
        Calculate the marginal for the combined set of variables1 and
        variables2 and adjust the variable indices

        Parameters:
        ----------
        variables1, variables2: set of integers

        Returns: joint_distribution, variable1_labels, variable2_labels
        -------
        joint_distribution: a numpy array
            Representing the marginal probability distribution for the 
            combined set of variables 1 and 2
        variable1_labels, variable2_labels: set of integers 
            The adjusted labels for variable 1 or 2 for the new joint 
            distribution

        """
        all_variables = variables1.union(variables2)
	joint_distribution = self.marginalize(all_variables, distribution)
        variable1_labels, variable2_labels = set(), set() 
        for count, variable in enumerate(sorted(list(all_variables))):
            if variable in variables1:
                variable1_labels.add(count)
            elif variable in variables2:
                variable2_labels.add(count)

        return joint_distribution, variable1_labels, variable2_labels

    def find_conditional_accounting_for_zero_marginals(
            self, marginal_variables, conditional_variables, 
            conditional_state_gen
            ):
	"""create the conditional distribution for the selected_indices given 
	the conditional_indices for the joint_distribution
	
	Parameters:
	----------
	marginal_indices: set of integers
            variables that are not conditioned on but are included
	conditional_indices: set of integers
            variables that are conditioned on
        conditional_state_gen: a generator
            Every time a marginal of the variables that are conditioned on
            is zero the generator is called for an "artificial" conditional
            state.
	
	Returns: conditional_distribution, marginal_labels, conditional_labels 
	-------
        conditional_distribution: a numpy array
        marginal_labels: list of integers
            The variables that are NOT conditioned on
        conditional_labels: list of integers
            The variables that are conditioned on

	"""
        joint_distribution, marginal_labels, conditional_labels = (
            self.find_joint_marginal(marginal_variables, conditional_variables)
        ) 
        
        marginal_conditional = self.marginalize(conditional_labels,
                                                joint_distribution)
        conditional_distribution = np.copy(joint_distribution)

        #first deal with the conditional values for zero marginals
        ix = np.argwhere(marginal_conditional==0)
        for element in ix:
            conditional_state = next(conditional_state_gen)
            conditional_distribution[tuple(element)] = conditional_state

        #set the rest of the values
        it = np.nditer(joint_distribution, flags=['multi_index'])
        while not it.finished:
            if it.value == 0:
                it.iternext()
                continue
            conditional_arguments = tuple(
                [it.multi_index[i] for i in conditional_labels]
            )
	    conditional_distribution[it.multi_index] = (
                it.value/marginal_conditional[conditional_arguments]
	    )
	    it.iternext()
	
        conditional_shape = [i for count, i in enumerate(joint_distribution.shape)
                             if count in conditional_labels]
        total_sum_conditional_distribution = reduce(lambda x,y: x*y, 
                                                    conditional_shape)
        if abs(np.sum(conditional_distribution)-total_sum_conditional_distribution)> 10**(-8):
            print("conditional distribution")
            print(conditional_distribution)
            raise ValueError("sum is {} while it should be {}".format(
                np.sum(conditional_distribution), total_sum_conditional_distribution
            ))

        return (conditional_distribution, marginal_labels, conditional_labels)

def compute_joint(marginal, conditional, conditional_labels):
    """compute the joint given the marginal and the conditional
    
    Parameters:
    ----------
    marginal: a numpy array
    conditional: a numpy array
    conditional_labels: a set of integers
        The length of conditional_labels must be equal to the number of 
        axis of marginal. In other words the variables of marginal should
        be equal to the variables that are conditioned on. The labels must
        be provided so that the order of the variables of the new joint and
        the old joint corresponds.

    """
    total_variables = len(conditional.shape)
    reordered_conditional = np.moveaxis(
        conditional, conditional_labels,
        range(total_variables-len(conditional_labels), total_variables, 1)
    )
    joint = reordered_conditional*marginal
    joint = np.moveaxis(
        joint, range(total_variables-len(conditional_labels), total_variables, 1), 
        conditional_labels
    )
    return joint 

def compute_joint_uniform_random(shape):
    """
    The joint distribution is generated by (uniform) randomly picking
    a point on the simplex given by (1, 1, ..., 1) with length the
    number of states of joint. In other words by sampling from the 
    Dirichlet(1, 1, 1, ..., 1)

    Parameters:
    ----------
    shape: a tuple of integers

    Returns:
    -------
    a numpy array with dimensions given by shape

    """
    number_of_states = reduce(lambda x,y: x*y, shape)
    dirichlet_random = np.random.dirichlet([1]*number_of_states)
    return np.reshape(dirichlet_random, shape)

def compute_joint_from_independent_marginals(marginal1, marginal2, marginal_labels):
    """
    Compute the joint using the marginals assuming independendence
    
    Parameters:
    ----------
    marginal1: a numpy array
        Representing an M-dimensional probability distribution
    marginal2: a numpy array
        Representing a probability distribution, the order of the variables
        should be in the same order as in the final joint
    marginal_labels: a sorted (from small to big) list of integers
        Representing on which axis the variables of marginal2 should be
        placed in the joint

    Returns: the joint distribution

    """
    outer_product = np.outer(marginal1, marginal2)
    joint = np.reshape(outer_product, marginal1.shape+marginal2.shape)
    for count, marginal_label in enumerate(marginal_labels):
        joint = np.rollaxis(joint, 
                            len(joint.shape)-len(marginal2.shape)+count,
                            marginal_label)
    return joint

def mutate_distribution_old(distribution, mutation_size):
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

def select_parents_old(amount_of_parents, sorted_population, rank_probabilities):
    """return the selected parents using stochastic universal selection 
    
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

def produce_distribution_with_entropy_evolutionary_old(
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
        parents = select_parents_old(number_of_children, sorted_population,
                                 rank_exp_probabilities)
        if i<number_of_trials/3.0:
            mutation_size = .3/total_number_of_states
        elif i<number_of_trials/2.0:
            mutation_size = 0.15/total_number_of_states
        elif i<(number_of_trials*0.75):
            mutation_size = 0.1/total_number_of_states
        else:
            mutation_size = 0.05/total_number_of_states
        children = [mutate_distribution_old(parent, mutation_size) for parent in parents]
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

def generate_probability_distribution_with_certain_entropy(
            shape, final_entropy_size,  
            zero_marginals_removed=False
            ):
    """
    Generate a probability distribution  with a certain entropy

    Parameters:
    ----------
    shape: tuple
    entropy_size: float

    Returns: a numpy array

    """
    total_number_of_states = reduce(lambda x,y: x*y, shape)
    max_difference = 1.0/total_number_of_states
    distribution = np.full(total_number_of_states, 1.0/total_number_of_states)
    entropy_size = entropy(distribution, base=2)
    #print("the initial entropy is {}".format(entropy_size))

    while entropy_size > final_entropy_size:
        #print('starting again')
        random_positions = np.random.choice(total_number_of_states, total_number_of_states*50*2)
        for i in range(total_number_of_states*50):
            if zero_marginals_removed: 
                entropy_size -= decrease_entropy(
                    distribution, random_positions[i],
                    random_positions[i*2], max_difference, set_zero=True 
                )
            else:
                entropy_size -= decrease_entropy(
                    distribution, random_positions[i],
                    random_positions[i*2], max_difference,  
                )
            if entropy_size < final_entropy_size:
                break 
            
    np.random.shuffle(distribution)
    distribution = np.reshape(distribution, shape)
    if zero_marginals_removed:
        distribution = remove_zero_marginals(distribution)
    return distribution

def remove_zero_marginals(distribution):
    """remove all states for which a marginal has probability zero
    
    Parameters:
    ----------
    distribution:

    Returns:
    -------
    updated distribution (a copy of the original distribution)

    """
    new_distribution = np.copy(distribution)
    probability_arr = ProbabilityArray(new_distribution)
    states_to_be_removed = {} 
    for variable in range(len(distribution.shape)):
        marginal = probability_arr.marginalize(set([variable]))
        zero_states = []
        for j, state in enumerate(marginal):
            if state == 0:
                zero_states.append(j)

        if zero_states != []:
            states_to_be_removed[variable] = zero_states
    
    for variable, states in states_to_be_removed.items():
        new_distribution = np.delete(new_distribution, states, variable)

    if abs(np.sum(new_distribution)-1) > 10**(-10):
        raise ValueError

    probability_arr = ProbabilityArray(new_distribution)
    for variable in range(len(new_distribution.shape)):
        if np.any(probability_arr.marginalize(set([variable]))==0):
            raise ValueError()

    return new_distribution

def decrease_entropy(distribution, state1, state2, max_difference, set_zero=False):
    """
    Decrease the entropy of a distribution by a random amount

    Parameters:
    ----------
    distribution: a 1-d numpy array 
        Representing a probability distribution
    state1, state2: integers in the range len(distribution)
    
    Returns: a float
    -------
    By how much the entropy was decreased

    """
    if distribution[state1]==0 or distribution[state2]==0:
        return 0
    elif distribution[state1] >= distribution[state2]:
        initial_entropy = np.dot(np.log2(distribution[[state1, state2]]),
                                 distribution[[state1, state2]])
        change = np.random.uniform(0, min(max_difference, distribution[state2]))
        if set_zero:
            if distribution[state2] < (10**(-10)):
                change = distribution[state2]

        distribution[state1] += change
        distribution[state2] -= change
        if distribution[state2] != 0:
            entropy_after = np.dot(np.log2(distribution[[state1, state2]]), 
                                   distribution[[state1, state2]])
        else:
            entropy_after = np.log2(distribution[state1]) * distribution[state1]
        return -(initial_entropy - entropy_after)
    else:
        return decrease_entropy(distribution, state2, state1, max_difference)

class ProbabilityDict():
    """
    Save a discrete probability distribution in a dictionary with keys
    representing states and values the probabilities
    
    """

    def __init__(self, probability_dict):
        """
        Parameters:
        ---------
        probability_dict: a dict
            The keys represent the different states (as tuples!) and 
            the values represent the probability of the state

        """
        self.probability_dict = probability_dict
        
    def print_distribution(self, sort=False):
        """
        Fancy printing method for distribution

        Parameters:
        ----------
        sort: boolean
            if sort is true the states will be printed in sorted order

        """

        if sort:
            prob_items = sorted(self.probability_dict.items(), 
                   lambda x, y: -1 if not x[0]>y[0] else 1)
        else:
            prob_items = self.probability_dict.items()
            
        for key, value in prob_items:
            print("{}: {}".format(key, value))
            
    def calculate_marginal_distribution(self, chosen_variable_indices):
        """ 
        Calculate the marginal distribution

        Parameters:
        ----------
       chosen_variable_indices: a set 
            variables for which the marginal will be calculated
        """

        marginal_distribution = {}
        for state, value in self.probability_dict.items():
            marginal_state = tuple(
                [entry for count, entry in enumerate(state) 
                 if count in chosen_variable_indices]
            )
            marginal_distribution[marginal_state] = value + marginal_distribution.get(marginal_state, 0)

        return marginal_distribution

    def calculate_entropy(self, variable_indices):
        """ 
        Calculate the entropy

        Parameters:
            variable_indices: a set
                All (variable) indices for which the entropy be calculated

        """
        distribution = self.calculate_marginal_distribution(variable_indices) 
        distribution_values_arr = np.array(distribution.values())
        distribution_values = distribution_values_arr[distribution_values_arr != 0]
        return - np.sum(np.log2(distribution_values) * distribution_values)

    def calulate_mutual_information(self, variable_indices1, variable_indices2):
        """
        Calculate the mutual information

        Parameters:
        ----------
        variable_indices1: a set
            All variable indices for the first distribution
        variable_indices2: All variable indices for second distribution

        """
        return (self.calculate_entropy(variable_indices1) +
                self.calculate_entropy(variable_indices2) -
                self.calculate_entropy(variable_indices1+variable_indices2))


