import numpy as np
from scipy.stats import entropy
import itertools
import copy
import random

from jointpdf.jointpdf import JointProbabilityMatrix
from jointpdf.jointpdf import FullNestedArrayOfProbabilities
from extension_probability_matrix import JointProbabilityMatrixExtended
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

def nudge_distribution_non_local_non_causal(
            joint, nudge_labels, nudge_size, nudge_option='random'
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
    nudge_option: string in set {random, proportional}
        see mutate_array_bigger_zero option docs
    
    Returns: a numpy array, representing the nudged probability distribution
    
    """
    nudged_joint = np.copy(joint)
    nudged_indices = tuple(range(len(joint.shape)-len(nudge_labels), len(joint.shape), 1))
    nudged_joint = np.moveaxis(nudged_joint, nudge_labels, nudged_indices)
    nudged_size_per_state = np.sum(nudged_joint, axis=nudged_indices)*nudge_size
    it = np.nditer(nudged_size_per_state, flags=['multi_index'])
    while not it.finished:
        if it.value == 0:
            it.iternext()
            continue

        flattened_dist = nudged_joint[it.multi_index].flatten()
        nudged_state = np.reshape(
            mutate_array_bigger_zero(flattened_dist, it.value, nudge_option),
            nudged_joint[it.multi_index].shape
        )
        nudged_joint[it.multi_index] = nudged_state 
        it.iternext()

    nudged_joint = np.moveaxis(nudged_joint, nudged_indices, nudge_labels)
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
    minus_states = select_subset(nudged_array, nudge_size)
    minus_mask = minus_states==1
    plus_mask = minus_states==0
    if np.sum(nudged_array[minus_mask]) < nudge_size:
        raise ValueError("chosen minus states wrongly")

    if option == 'proportional': 
        minus_part = nudged_array[minus_mask]
        minus_nudge = (minus_part/np.sum(minus_part)) * nudge_size 
        plus_part = nudged_array[plus_mask]
        plus_nudge = (plus_part/np.sum(plus_part)) * nudge_size 
    elif option == 'random':
        minus_nudge = nudge_size * np.random.dirichlet(
            [1]*np.count_nonzero(minus_states)
        )
        minus_nudge = np.minimum(nudged_array[minus_mask], minus_nudge)
        difference = abs(np.sum(minus_nudge)-nudge_size)
        count = 0
        while difference > 10**(-10) and count<10:
            count += 1
            number_of_free_states = minus_nudge[nudged_array[minus_mask]!=minus_nudge].shape[0]
            redistribute = difference * np.random.dirichlet([1]*number_of_free_states)
            minus_nudge[nudged_array[minus_mask]!=minus_nudge] += redistribute
            minus_nudge = np.minimum(nudged_array[minus_mask], minus_nudge)
            difference = abs(np.sum(minus_nudge)-nudge_size)
        if count == 10:
            print("couldnt find nudge totally randomly, now proportional")
            free_space = nudged_array[minus_mask] - minus_nudge
            minus_nudge += (free_space/np.sum(free_space))*difference

        if abs(np.sum(minus_nudge)-nudge_size)> 10**(-8):
            raise ValueError("minus nudge not big enough")       
        minus_nudge = minus_nudge
        plus_nudge = nudge_size * np.random.dirichlet(
            [1]*(minus_states.shape[0]-np.count_nonzero(minus_states))
        )

    nudged_array[minus_mask] = nudged_array[minus_mask] - minus_nudge
    nudged_array[plus_mask] = nudged_array[plus_mask] + plus_nudge 
    return nudged_array

def find_max_global_impact(input_dist, cond_output, nudge_size):
    """
    Find the maximum nudge output, defined as the abs difference between
    the old and the new output state.

    Parameters:
    ----------
    input_dist: a numpy array
        Representing the input distribution
    cond_output: a numpy array
        Representing the conditional probabilities of the output given 
        the input. The first N-1 variables (axes) should represent the
        input variables while the output should be the last axis.
    nudge_size: A number

    Returns: nudge, nudge_size
    -------
    nudge: a list of tuples
        The nudge that gives the maximum impact.
        Every tuple consist of an integer representing the nudge state on 
        the flattened input array and a number representing the nudge size
        on this state.
    nudge_size: The impact size of this nudge, defined as the max difference

    """
    number_of_input_vars = input_dist.flatten().shape[0]
    max_impact, max_nudge_states, max_nudge_sizes = 0, [], []
    for output_nudge_state in itertools.product([-1, 1], repeat=cond_output.shape[-1]):
        if all([state==-1 for state in output_nudge_state]) or all([state==1 for state in output_nudge_state]):
            continue

        allignment = cond_output.flatten() * np.array(output_nudge_state*number_of_input_vars)
        allignment_scores = np.sum(
            np.reshape(allignment, (number_of_input_vars, len(output_nudge_state))),
            axis=1
        )
        if output_nudge_state == (-1, -1, 1):
            print(allignment_scores)

        #print(output_nudge_state)
        nudge_states, nudge_sizes, nudge_impact = find_nudge_states_and_sizes(
            allignment_scores, input_dist.flatten(), nudge_size
        )
        #print("nudge states {}".format(nudge_states))
        #print("nudge sizes {}".format(nudge_sizes))
        #print("nudge_impact {}".format(nudge_impact))

        if nudge_impact > max_impact:
            max_impact = nudge_impact
            max_nudge_states, max_nudge_sizes = nudge_states, nudge_sizes

    return max_nudge_states, max_nudge_sizes, max_impact

#not happy with the approach taken here!
def find_max_local_impact(input_dist, cond_output, nudge_size, nudge_states):
    """
    Find the max local nudge impact. The local nudge is performed on
    the marginal distribution only and all input states in joint input
    distribution distribution are affected proportionally.

    Parameters:
    ----------
    input_dist: nd-array
        A probability distribution (see ProbabilityArray class)
    cond_output: nd-array
        The distribution of the output conditioned on the input. The 
        output should be on the last axis.
    nudge_size: a number
    nudge_states: a list of integers

    Returns:
    -------
    max_nudge: A list of tuples 
        The maximum nudge that can be performed. The list has the same
        length as the flattened input distribution of only the nudged
        states. every tuple is a pair of a state and a nudge size.
    max_impact: a number
        The max absolute difference that can be achieved in the
        output distribtuion by a nudge of nudge size. 

    """
    number_of_input_states = input_arr.flatten().shape[0]
    input_variables = list(range(len(input_dist.shape)))
    new_nudged_states = input_variables[:len(nudged_variables)]
    new_non_nudged_states = input_variables[len(nudged_variables):]
    input_arr = np.moveaxis(input_dist, nudge_states, new_nudged_states)
    scores = np.moveaxis(cond_output, nudge_states, new_nudged_states)

    weights = np.sum(input_arr, axis=tuple(new_non_nudged_states))
    weighted_scores = np.multiply(
        np.repeat(input_arr.flatten(), cond_output.shape[-1]),
        scores.flatten()
    )

    max_impact, max_nudge_states, max_nudge_sizes = 0, [], []
    all_allignments_output_states = itertools.product([-1, 1], repeat=cond_output.shape[-1])
    redundant_states = [[i]*len(allignment_output_state) for i in [-1, 1]]
    for allignment_output_state in all_allignments_output_states:
        if allignment_output_state in redundant_states:
            continue
            
        print(allignment_output_state)
        allignment = weighted_scores * np.array(allignment_output_state*number_of_input_states)
        allignment_scores = np.sum(
            np.reshape(allignment, cond_output.shape),
            axis=tuple(new_non_nudged_states)
        )
        nudge_states, nudge_sizes, nudge_impact = find_nudge_states_and_sizes(
            allignment_scores, weights, nudge_size
        )
        if nudge_impact > max_impact:
            max_impact = nudge_impact
            max_nudge_states, max_nudge_size = nudge_states, nudge_sizes

    return max_nudge_states, max_nudge_sizes, max_impact

def find_nudge_states_and_sizes(scores, weights, nudge_size):
    """
    Find the nudge that pushes the output variable maximally towards the 
    output state

    Parameters:
    ----------
    scores: 1-D nd-array
        The alligment scores for the conditional distribution linked to
        all input states
    weights: nd-array
       The flattened input probabilities
    nudge_size: a number

    Returns: nudge, nudge_impact
    -------
    selected states: list
        The states that will be nudged
    nudge_sizes: nd-array
        The nudge size for the corresponding selected state
    nudge_impact: a number
        The maximum that that the output can be transformed towards the
        output state by changing the input state with the nudge size 

    """
    negative_states, negative_nudge_sizes = find_minimum_subset(
        weights, scores, nudge_size
    )
    positive_state = [np.argmax(scores)]
    positive_nudge_size = sum(negative_nudge_sizes)
    selected_states = negative_states + positive_state
    nudge_sizes = np.zeros(len(selected_states))
    nudge_sizes[:-1] = -1*negative_nudge_sizes
    nudge_sizes[-1] = positive_nudge_size
    nudge_impact = np.sum(scores[selected_states] * nudge_sizes)
    return selected_states, nudge_sizes, nudge_impact

def find_minimum_subset(weights, scores, total_size):
    """
    Find a subset of weights of total_size with the lowest sum in scores.
    Fractions of scores CAN be taken.

    Note: The selected subset cannot include all of the weights

    Parameters:
    ----------
    weights: a 1d nd-array
    scores: a 1d nd-array with the same length of weights
    total_size: a number

    Returns: selected_variables, selected_weights, total_impact
    -------
    selected_variables: list
    selected_weights: nd-array 
        The corresponding weights for the selected_variables

    """
    #can (probably) be done more efficient for now fine
    indices = list(range(weights.shape[0]))
    results = [[weight, score, index] for weight, score, index
              in zip(weights, scores, indices)]
    results_sorted = sorted(results, key=lambda x: x[1])
    weights_sorted, scores_sorted, indices_sorted = zip(*results_sorted)

    count = 0
    minus_weight = 0
    while minus_weight < total_size and count < len(indices)-1:
        minus_weight += weights_sorted[count]
        count += 1
    
    selected_indices = list(indices_sorted[:count])
    selected_weights = weights[selected_indices]
    selected_weights[-1] = min(total_size-sum(selected_weights[:-1]),
                               selected_weights[-1])
    return selected_indices, selected_weights

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

def impact_nudge_causal_output(
        distribution, function_indices, new_input_distribution,
        use_l1norm=False
        ):
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
    if use_l1norm:
        return np.sum(np.absolute(marginal_output_old-marginal_output_new))
    else:
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


