import numpy as np
from scipy.stats import entropy
import itertools
import copy
import random

from jointpdf.jointpdf import JointProbabilityMatrix
from jointpdf.jointpdf import FullNestedArrayOfProbabilities
from probability_distributions import JointProbabilityMatrixExtended
import probability_distributions
from probability_distributions import ProbabilityArray

def nudge(distribution, nudge_size):
    """
    Perform a nudge on the distribution. For now the nudge is performed
    on two randomly selected states
    
    Parameters:
    ----------
    distribution: a 1-d numpy array
    nudge_size:  a positive number
        The size of the nudge
    
    Returns: nudged_distribution, nudged_states
    -------
    nudged_distribution: a numpy array
    nudges_states: a numpy array
        The two states that are nudged

    """
    nudged_states = np.random.choice(distribution.shape[0],
                                     size=2, 
                                     replace=False)
    nudge_size = min(nudge_size, 
                     1-distribution[nudged_states[0]],
                     distribution[nudged_states[1]])
    
    nudged_distribution = np.copy(distribution)
    nudged_distribution[nudged_states[0]] += nudge_size
    nudged_distribution[nudged_states[1]] -= nudge_size
    return nudged_distribution, nudged_states

def effect_of_nudge_1d(distribution, nudge_size):
    """
    Nudge the input variable and calculate the effect on the output variable
    (the KL-devergence of the output variable)
    
    Parameters:
    ----------
    distribution: a numpy array
        It should represent the joint probability distribution of 1 input
        (the first axis) and 1 output variable (the second axis).
    nudge_size: a number
    
    Returns: a number
    """
    probability_array_old = ProbabilityArray(distribution)
    marginal_variable_old = probability_array_old.marginalize(set([0]))
    marginal_function_old = probability_array_old.marginalize(set([1]))
    conditional_joint_old, marginal_labels_old, conditional_labels_old = (
        probability_array_old.find_conditional(set([1]), set([0]))
    )
    marginal_variable_nudged = nudge(marginal_variable_old, NUDGE_SIZE)
    joint_new = ProbabilityArray(probability_distributions.compute_joint(
        marginal_variable_nudged, conditional_joint_old, conditional_labels_old
    ))
    marginal_function_new = joint_new.marginalize(set([1]))  
    kl_variable = entropy(marginal_variable_old, marginal_variable_nudged)
    kl_function = entropy(marginal_function_old, marginal_function_new) 
    return kl_variable, kl_function

def select_random_states(statespace, number_of_samples):
    """
    select randomly the number of samples from the statespace

    Parameters:
    ----------
    statespace: an iterable of integers
        Every integer represents the number of states of the variables
        with that index.
    number_of_samples: integer

    Returns: states
    -------
    states: a list of lists
        Every list represents one state

    """
    states = []
    for i in range(number_of_samples):
        state = []
        for number_of_states in statespace:
            state.append(np.random.randint(number_of_states))

        states.append(state)

    return states

def nudge_distribution_local_non_causal(joint, nudge_label, nudge_size, number_of_nudges):
    """
    Nudge the the variable with nudge label while keeping the 
    marginal of the other variables constant. Thus assuming that the variable
    on which the nudge is performed does not causally impact the other variables.
    Moreover, the nudge moves probability weight from one state of the marginal
    of the nudged variable to another state of the marginal of the nudged variable.
    
    Parameters:
    ----------
    joint: a numpy array
        Representing a discrete probability distribution
    nudge_label: an integer
    nudge_size: a (small) number
    number_of_nudges: an integer
    
    Returns: a numpy array, representing the nudged probability distribution
    
    """
    nudged_joint = np.copy(joint)
    nudged_joint = nudged_joint.swapaxes(nudge_label, len(joint.shape)-1)
    nudge_states = select_random_states(nudged_joint.shape[:-1], number_of_nudges) 
    
    nudged_states_marginal = np.random.choice(joint.shape[nudge_label], 2, replace=False)
    nudge_state_plus, nudge_state_minus = nudged_states_marginal[0], nudged_states_marginal[1]   
    for state in nudge_states:
        plus_state = tuple(copy.copy(state) + [nudge_state_plus])
        minus_state = tuple(copy.copy(state) + [nudge_state_minus])        
        size = min(nudged_joint[minus_state], 1-nudged_joint[plus_state], nudge_size)
        nudged_joint[plus_state] += size
        nudged_joint[minus_state] -= size
    
    nudged_joint = nudged_joint.swapaxes(nudge_label, len(joint.shape)-1)
    return nudged_joint

def nudge_distribution_local_non_causal_multiple_variables(
            joint, nudge_labels, nudge_size
            ):
    """
    Nudge the the variable with nudge label while keeping the 
    marginal of the other variables constant. Thus assuming that the variable
    on which the nudge is performed does not causally impact the other variables.
    The nudge moves weight around in a random manner. Meaning that the weight 
    does not go from one state to another state, but rather a random 
    perturbation vector is placed on the states, its sum being equal
    to 0 and its absolute sum equal to 2*nudge_size
    
    Parameters:
    ----------
    joint: a numpy array
        Representing a discrete probability distribution
    nudge_labels: a list of integers
    nudge_size: a (small) number
    
    Returns: a numpy array, representing the nudged probability distribution
    
    """
    nudged_joint = np.copy(joint)
    nudged_indices = tuple(range(len(joint.shape)-len(nudge_labels), len(joint.shape), 1))
    nudged_joint = nudged_joint.moveaxis(nudge_labels, nudged_indices)
    nudged_size_per_state = np.sum(nudged_joint, axis=nudged_indices)*nudge_size
    it = np.nditer(nudged_size_per_state, flags=['multi_index'])
    while not it.finished:
        if it.value == 0:
            it.iternext()
            continue

        shape = nudged_joint[it.multi_index].shape
        flattened_dist = np.flatten(nudged_joint[it.multi_index])
        nudged_state = mutate_array_bigger_zero(
            nudged_joint[it.multi_index], it.value
        )
        nudged_joint[it.multi_index] = np.reshape(nudged_state, shape) 
        it.iternext()

    nudged_joint = nudged_joint.moveaxis(nudge_labels, nudged_indices)
    return nudged_joint

def mutate_array_bigger_zero(arr, nudge_size, option):
    """ 
    Mutate the arr under the constraint that every entry must be 
    bigger or equal to zero. The total plus and minus change should 
    both be equal to nudge size

    Parameters:
    ----------
    arr: a 1-d nd array
    nudge_size: A (small) number
    option: string in set: {proportional, random} 
        how to create the vector to perform the nudge. With proportional
        the plus and minus states are chosen randomly (within the 
        constraint that the minus states must have weight bigger than
        nudge size). Then the nudge size in the plus and minus states are 
        totally proportional to the old probability masses in those states.
        For random the weights for the minus states are selected using the 
        Dirichlet distribution, if this pushes the state under zero than, 
        that part of the nudge is redistributed among the other states (
        again using Dirichlet)

    Returns: a 1-d numpy array

    """
    nudged_array = np.copy(arr)
    minus_states = select_subset(nudged_array)
    plus_states = np.array([1 if i==0 else 0 for i in minus_states])
    if option == 'proportional': 
        minus_part = nudged_array[plus_states]
        minus_nudge = (minus_part/np.sum(minus_part)) * it.value
        plus_part = nudged_array[minus_states]
        plus_nudge = (plus_part/np.sum(plus_part)) * it.value
    elif option == 'random':
        proposed_minus_nudge = np.random.dirichlet([nudge_size]*minus_states.shape[0])
        actual_minus_nudge = np.minimum(nudged_array[plus_states], proposed_minus_nudge)
        difference = abs(np.sum(actual_minus_nudge)-nudge_size)
        while difference > 10**(-8):
            mask = nudged_array[minus_states]!=actual_minus_nudge
            proposal_redistribution = np.random.dirichlet([difference]*np.count_nonzero(mask))
            proposal_minus_nudge[mask] = actual_minus_nudge[mask] + proposal_redistribution
            actual_minus_nudge = np.minimum(nudged_array[plus_states], proposed_minus_nudge)
            difference = abs(np.sum(actual_minus_nudge)-nudge_size)

        minus_nudge = actual_minus_nudge
        plus_nudge = np.random.dirichlet([nudge_size]*plus_states.shape[0])

    nudged_array[minus_states] = nudged_array[minus_states] - minus_nudge
    nudged_array[plus_states] = nudged_array[plus_states] + plus_nudge 
    return nudged_array

def select_subset(arr, threshold):
    """
    Select a subset of the arr (randomly) which is at least bigger than
    threshold
    
    Parameters:
    ----------
    arr: a 1-d numpy array
        All entries should be greater or equal to zero

    Returns: a 1-d numpy filled with zeros and ones

    """
    minus_states = np.random.randint(0, 2, arr.shape[0])
    attempt_count = 0
    while (np.sum(arr[minus_states!=0])<threshold or np.all(minus_states==1)) and attempt_count<20:
        minus_states = np.random.randint(0, 2, arr.shape[0])
        attempt_count += 1

    if attempt_count > 19:
        print("could not find satisfying minus state randomly")
        minus_states = np.ones(arr.shape[0])
        indices = list(range(arr.shape[0]))
        np.random.shuffle(indices)
        for index in indices:
            if arr[minus_states]-arr[index] > threshold:
                minus_states[selected] = 0

    return minus_states

def impact_nudge_causal_output(distribution, function_indices, new_input_distribution):
    """
    Calculate the impact of a nudge of the input distribution on the output. 
    Assuming the output is causally determined using using the input.
    
    Parameters:
    ----------
    distribution: a ProbabilityArray object
    function_indices: a set of integers
    new_input_distribution: a numpy array
        It represents the input distribution after the nudge
    
    Returns:
    -------
    A numpy array representing a probability distribution
    
    """
    variable_indices = set(range(len(distribution.probability_distribution.shape))) - function_indices
    marginal_output_old = distribution.marginalize(function_indices)
    conditional, marginal_labels, conditional_labels = (
        distribution.find_conditional(function_indices, variable_indices)
    )
    distribution_new = ProbabilityArray(probability_distributions.compute_joint(
        new_input_distribution, conditional, conditional_labels
    ))
    marginal_output_new = distribution_new.marginalize(function_indices)  
    kl_divergence = entropy(marginal_output_old, marginal_output_new) 
    return kl_divergence
        
def perform_nudge(distribution, minus_state, plus_state, proposed_nudge):
    """
    Perform the nudge on the states of the distribution within the 
    constraints that every entry is >= 0 and <= 1.

    Parameters:
    ----------
    distribution: a numpy array
        Representing a probability density function
    minus_state, plus_state: tuples
    proposed_nudge: number

    Returns: a number
        The performed nudge

    """
    plus_probability = distribution[plus_state]
    minus_probability = distribution[minus_state]
    nudge_size = min(minus_probability, 1-plus_probability, proposed_nudge)
    distribution[plus_state] += nudge_size
    distribution[minus_state] -= nudge_size 
    return nudge_size

def mutate_distribution_with_fixed_marginals(distribution, output_label, 
                                             amount_of_mutations, nudge_size):
    """
    Mutate the joint distribution while keeping the marginals of the input
    and output constant

    Parameters:
    ----------
    distribution: a numpy array
        Representing a probability distribution
    output_label: integer
    amount_of_mutations: The amount of mutations to be applied
    nudge_size: the average nudge size to be applied

    Returns: a numpy array, representing the probability distribution

    """
    number_of_variables = len(distribution.shape)
    mutated_distribution = np.copy(distribution)
    mutated_distribution = np.swapaxes(mutated_distribution, output_label, 
                                       number_of_variables-1)
    
    states = select_random_states(mutated_distribution.shape[:-1],
                                  amount_of_mutations)
    mutation_states = np.random.randint(0, mutated_distribution.shape[-1],
                                        (amount_of_mutations, 2))
    plus_states = np.zeros((amount_of_mutations, number_of_variables), np.int32)
    plus_states[:, :-1] = np.array(states)
    plus_states[:, -1] = mutation_states[:, 0]
    minus_states = np.zeros((amount_of_mutations, number_of_variables), np.int32)
    minus_states[:, :-1] = np.array(states)
    minus_states[:, -1] = mutation_states[:, 1]

    probability_mass_change = {k:0 for k in range(mutated_distribution.shape[-1])} 
    for i in range(amount_of_mutations):
        proposed_nudge = max(np.random.normal(nudge_size, 0.2*nudge_size), 0)
        performed_nudge_size = perform_nudge(
            mutated_distribution, tuple(minus_states[i]), 
            tuple(plus_states[i]), proposed_nudge
        )
        probability_mass_change[mutation_states[i, 0]] += performed_nudge_size
        probability_mass_change[mutation_states[i, 1]] -= performed_nudge_size

    if abs(np.sum(probability_mass_change.values())) > 10**-6:
        raise ValueError("changes do not cancel!")

    #print("performed initial nudge")
    variable_list = [item[0] for item in 
                     sorted(probability_mass_change.items(), key=lambda x: x[1])]

    plus_state = np.zeros(number_of_variables, np.int32)
    minus_state = np.zeros(number_of_variables, np.int32)
    while variable_list != []:
        variable_items = [(k, v) for k, v in probability_mass_change.items() 
                          if k in variable_list]
        minimum_index, _ = min(variable_items, key=lambda x: x[1])
        variable_list.remove(minimum_index)
        variable = minimum_index
        plus_state[-1] = variable
        count2 = 0
        while abs(probability_mass_change[variable]) > 10**-6 and count2<500:
            count2 += 1
            #print(probability_mass_change)
            state = select_random_states(mutated_distribution.shape[:-1], 1)[0] 
            plus_state[:-1], minus_state[:-1] = state, state
            minus_state[-1] = int(np.random.choice(variable_list))
            #print("the minus state is {}, prob {}".format(minus_state, mutated_distribution[tuple(minus_state)]))
            #print("the plus state is {} prob {}".format(plus_state, mutated_distribution[tuple(plus_state)]))
            #print("the probability mass {}".format(probability_mass_change[variable]))
            #print("nudge size {}, proposed nudge {}".format(nudge_size, min(nudge_size, abs(probability_mass_change[variable]))))
            performed_nudge_size = perform_nudge(
                mutated_distribution, tuple(minus_state), tuple(plus_state),
                min(nudge_size, abs(probability_mass_change[variable]))    
            )
            #print("the performed nudge size is {}".format(performed_nudge_size))
            probability_mass_change[variable] += performed_nudge_size
            probability_mass_change[minus_state[-1]] -= performed_nudge_size

    mutated_distribution = np.swapaxes(mutated_distribution, output_label, 
                                       len(distribution.shape)-1)
    return mutated_distribution


