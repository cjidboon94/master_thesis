import numpy as np
from scipy import stats
from probability_distributions import ProbabilityArray
import probability_distributions
import copy

np.seterr(all='warn')

def find_nudge_impact(old_input, new_input, conditional_output, measure="absolute"):
    """
    Find the impact of a nudge transforming the old input into
    the new input.
    
    Parameters:
    ----------
    old_input: nd-array
        representing a probability distribution
    new_input: nd-array
        representing a probability distribution
    conditional_output: nd-array
        represening the probability distribution, the last variable
        should be the conditional output (in tree form the leaves should be
        the conditional output)
    measure: string
        Should be in: {"absolute", "kl-divergence"}
        
    Returns: a number
    
    """
    number_of_input_vars = len(old_input.shape)
    old_joint = probability_distributions.compute_joint(
        old_input, conditional_output, set(range(0, number_of_input_vars, 1))
    )
    #print("old joint {}".format(old_joint))
    old_output = ProbabilityArray(old_joint).marginalize(set([number_of_input_vars]))
    #print("old output {}".format(old_output))
    new_joint = probability_distributions.compute_joint(
        new_input, conditional_output, set(range(0, number_of_input_vars, 1))
    )
    new_output = ProbabilityArray(new_joint).marginalize(set([number_of_input_vars]))
    #print("new output {}".format(new_output))
    if measure=="absolute":
        return np.sum(np.absolute(old_output.flatten()-new_output.flatten()))
    elif measure=="kl-divergence":
        return stats.entropy(old_output.flatten(), new_output.flatten(), base=2)
    else:
        raise ValueError("provide a valid measure")

def nudge_global(dist, nudge_size, without_conditional=False):
    if not without_conditional:
        return np.reshape(
            nudge_individual(dist.flatten(), nudge_size),
            dist.shape
        )
    else:
        return np.reshape(
            nudge_individual_without_conditional(dist.flatten(), nudge_size),
            dist.shape
        )

def nudge_local(dist, nudged_vars, nudge_size, noise_vectors=None,
                without_conditional=False):
    """perform multiple individual nudges

    Parameters:
    ----------
    dist: a numpy array 
        Representing a distribution, the last variable is the one
        that will be nudged
    nudged_vars: a list of ints
        Representing all the variables that will be nudged
    nudge_size: a number
    noise_vectors: a list of 1-d nd-arrays
        Every entry is a noise vector to nudge the distribution

    """
    if len(dist.shape) == 1:
        raise ValueError("cannot do local nudge on 1-d dist, use individual")
    old_dist = dist
    new_dist = np.copy(dist)
    total_number_of_vars = len(dist.shape)
    total_nudge = np.zeros(old_dist.shape)
    for count, var in enumerate(nudged_vars):
        #print("the var that will be nudged {}".format(var))
        #print("the last var {}".format(total_number_of_vars))
        #print("dist shape {}".format(old_dist.shape))
        old_dist = np.swapaxes(old_dist, var, total_number_of_vars-1)
        if noise_vectors is None and not without_conditional:
            new_dist = nudge_individual(old_dist, nudge_size)
        elif noise_vectors is None and without_conditional:
            new_dist = nudge_individual_without_conditional(old_dist, nudge_size)
        elif noise_vectors is not None and not without_conditional:
            new_dist = nudge_individual(old_dist, nudge_size,
                                        noise_vectors[count])
        elif noise_vectors is not None and without_conditional:
            new_dist = nudge_individual_without_conditional(
                old_dist, nudge_size, noise_vectors[count]
            )

        nudge = new_dist-old_dist
        if abs(np.sum(nudge)) > 10**(-10):
            raise ValueError("the nudge should sum to zero")
        old_dist = np.swapaxes(new_dist, var, total_number_of_vars-1)
        nudge = np.swapaxes(nudge, var, total_number_of_vars-1)
        total_nudge += nudge

    if abs(np.sum(total_nudge)) > 10**(-10):
        raise ValueError("total nudge should sum sum to zero")
    total_nudge_size = np.sum(np.absolute(total_nudge))
    #print("the total nudge size {}".format(total_nudge_size))
    #print("unnormalised second marginal {}".format(
    #    ProbabilityArray(dist+total_nudge).marginalize(set([1]))
    #))
    nudge_vector = total_nudge * (nudge_size/total_nudge_size)
    new_dist = nudge_states(nudge_vector.flatten(), np.copy(dist).flatten())
    new_dist = np.reshape(new_dist, dist.shape)
    return new_dist

def nudge_individual(dist, nudge_size, noise_vector=None):
    """
    Perform a local non causal nudge. The function does not change the
    old distribution.
    
    Parameters:
    ----------
    dist: a numpy array 
        Representing a distribution, the last variable is the one
        that will be nudged
    nudge_size: a number
    noise_vector: a 1-d nd-array
        The noise vector that is used to nudge the distribution.

    Returns: an nd-array (the new distribution)
    -------
    The  returned  array is new object, so 

    """
    number_of_vars = len(dist.shape)
    number_of_nudge_states = dist.shape[-1]
    if noise_vector is None:
        noise_vector = find_noise_vector(number_of_nudge_states, nudge_size)
    #print("the noise vector {}".format(noise_vector))
    #print("the l1-norm of the noise vector is {}".format(
    #    np.sum(np.absolute(noise_vector))
    #))
    if number_of_vars == 1:
        return nudge_states(noise_vector, dist)

    other_vars = np.sum(dist, axis=number_of_vars-1)
    other_vars = other_vars.flatten()
    number_of_other_states = other_vars.flatten().shape[0]
    cond_nudge, _, _ = ProbabilityArray(dist).find_conditional(
        set([number_of_vars-1]), set(range(number_of_vars-1))
    )
    cond_nudge = np.reshape(
        cond_nudge, (number_of_other_states, number_of_nudge_states)
    )
    #print("other vars {}".format(other_vars))
    #print("conditional nudge {}".format(cond_nudge))

    new_dist = np.zeros((number_of_other_states, number_of_nudge_states))
    for i in range(number_of_other_states):
        nudged_states = nudge_states(noise_vector, cond_nudge[i])
        #print("the nudge_states are {}".format(nudged_states))
        new_dist[i] = other_vars[i] * nudged_states

    new_dist = np.reshape(new_dist, dist.shape)
    return new_dist

def nudge_individual_without_conditional(dist, nudge_size, noise_vector=None):
    """Perform a local non causal nudge
    
    Parameters:
    ----------
    dist: a numpy array 
        Representing a distribution, the last variable is the one
        that will be nudged
    nudge_size: a number
    noise_vector: a nd-array defaults to None

    """
    number_of_vars = len(dist.shape)
    number_of_nudge_states = dist.shape[-1]
    if noise_vector is None:
        noise_vector = find_noise_vector(number_of_nudge_states, nudge_size)
    #print("the noise vector {}".format(noise_vector))
    if number_of_vars == 1:
        return nudge_states(noise_vector, dist)

    other_vars = np.sum(dist, axis=number_of_vars-1)
    other_vars = other_vars.flatten()
    number_of_other_states = other_vars.flatten().shape[0]
    dist_reshaped = np.reshape(
        np.copy(dist), (number_of_other_states, number_of_nudge_states)
    )
    #print("other vars {}".format(other_vars))

    new_dist = np.zeros((number_of_other_states, number_of_nudge_states))
    for i in range(number_of_other_states):
        if other_vars[i] == 0:
            new_dist[i] = 0
            continue

        nudged_states = nudge_states(noise_vector, dist_reshaped[i]/other_vars[i])
        #print("the nudge_states are {}".format(nudged_states))
        new_dist[i] = other_vars[i] * nudged_states

    new_dist = np.reshape(new_dist, dist.shape)
    return new_dist

def find_noise_vector(number_of_states, noise_size):
    """
    find a noise vector

    Parameters:
    ----------
    number_of_states: a number
    noise_size: a number

    Returns: a numpy array
        It has size number of states and every state represents the 
        noise applied to these states. The sum equals zero and the sum
        of absolute numbers equals the noise_size

    """
    noise_vector = np.zeros(number_of_states)
    negative_states, positive_states = [], []
    while negative_states == [] or positive_states == []:
        negative_states, positive_states = [], []
        for state in np.arange(number_of_states):
            if np.random.random() > 0.5:
                negative_states.append(state)
            else:
                positive_states.append(state)  
                               
    #print("the positive states {}, negative states {}".format(
    #    positive_states, negative_states
    #))
    noise_vector[negative_states] = (
        (-noise_size/2) * np.random.dirichlet([1]*len(negative_states))
    )
    noise_vector[positive_states] = (
        (noise_size/2) * np.random.dirichlet([1]*len(positive_states))
    )
    return noise_vector

def nudge_states(noise, probabilities):
    """
    Find the new states after applying the noise vector.

    Parameters:
    ----------
    noise: a 1d numpy array
        It should sum to zero
    probabilities: a 1d numpy array
    
    Returns: a numpy array

    """
    if np.amin(probabilities+noise) >= 0:
        return probabilities+noise

    capped_noise = np.zeros(noise.shape)
    negative_states = noise<=0
    positive_states = noise>0
    capped_noise[negative_states] = -1 * np.minimum(
        np.absolute(noise[negative_states]), 
        np.absolute(probabilities[negative_states])
    )
    #print("the capped noise {}".format(capped_noise))
    if np.any(capped_noise[positive_states]):
        print("the noise {}".format(noise))
        print("the probabilities {}".format(probabilities))

    capped_noise[positive_states] = (
        (np.sum(capped_noise[negative_states])/np.sum(noise[negative_states]))
        * noise[positive_states]
    )
    if np.any(np.isnan(capped_noise[positive_states])):
        print(capped_noise[positive_states])
        print("the negative states {}".format(negative_states))
        print("sum of the negative states {}".format(np.sum(noise[negative_states])))
        print("the noise {}".format(noise))
        print("the probabilities {}".format(probabilities))
        raise ValueError("sum to 0")

    #print("the capped noise {}".format(capped_noise))
    return probabilities+capped_noise

def synergistic_mutate(distribution, max_mutation_size=None, mutation_dict=None, 
                       mutation_size=None):
    """
    Mutate the joint distribution without changing any of the marginals.

    Parameters:
    ----------
    distribution: nd-array representing probability distribution
    max_mutation_size: number
    mutation_dict: a dict with negative_states and positive_states where
        both values are a list of tuples represesenting which states to 
        mutate
    mutation_size: a positive number

    Returns: nd-array, 
    -------
    The mutated distribution which is a probability distribution itself.

    Note: 
    ----
    The distribution should have at least two variables and the mutation
    happens in place. Need to supply either max_mutation_size or 
    mutation_size. The method won't work well for distributions with a low
    entropy.

    """
    number_of_variables = len(distribution.shape)
    if number_of_variables < 2:
        raise ValueError("should be at least two variables")
    if mutation_dict is None:
        mutation_dict = select_random_synergistic_mutation_states(distribution)
    
    negative_states = mutation_dict["negative_states"]
    positive_states = mutation_dict["positive_states"]

    #what to do if any of the negative mutation states is smaller than the
    #mutation_size -simply set the mutation to zero? this will push the
    #distribution to uniform (after many mutations) 
    #solution is to maximize the mutation size to max of the weights of the states
    #plus some small number ...
    if mutation_size is not None:
        mutation_size = min(
            distribution[negative_states[0]],
            distribution[negative_states[1]],
            mutation_size
        )
    elif mutation_size is None:
        maximum_mutation_size = min(
            distribution[negative_states[0]],
            distribution[negative_states[1]],
            max_mutation_size
        )
        mutation_size = np.random.uniform(0, maximum_mutation_size)

    for state in negative_states:
        distribution[state] -= mutation_size
    for state in positive_states:
        distribution[state] += mutation_size

    if np.any(distribution<0):
        print("something went wrong")
        print("negative states {}".format(negative_states))
        print("positive_states {}".format(positive_states))
        print("the mutation size {}".format(mutation_size))
        print("the negative probs {}, {}".format(distribution[negative_states[0]], distribution[negative_states[1]]))
        print("the positive probs {}, {}".format(distribution[positive_states[0]], distribution[positive_states[1]]))
        print(distribution)
        raise ValueError()

    return distribution

def select_random_synergistic_mutation_states(distribution):
    """
    
    Parameters:
    ----------
    distribution: nd-array representing a probability distribution

    Returns: 
    -------
    A dict with keys positive_states, negative_states where both have
    a list of two tuples as values. The tuples represent states of the 
    distribution.

    """
    number_of_variables = len(distribution.shape)
    nudged_var1, nudged_var2 = np.random.choice(
        number_of_variables, 2, replace=False
    )
    states_var1 = np.random.choice(
        distribution.shape[nudged_var1], 2, replace=False
    )
    states_var2 = np.random.choice(
        distribution.shape[nudged_var2], 2, replace=False
    )
    other_vars = list(range(number_of_variables))
    other_states = [np.random.choice(distribution.shape[other_vars[var]])
                    for var in other_vars]

    #print(other_states)
    states1, states2, states3, states4 = (
        copy.copy(other_states), copy.copy(other_states), 
        copy.copy(other_states), copy.copy(other_states)
    )
    states1[nudged_var1] = states_var1[0]
    states1[nudged_var2] = states_var2[0]
    states2[nudged_var1] = states_var1[0]
    states2[nudged_var2] = states_var2[1]
    states3[nudged_var1] = states_var1[1]
    states3[nudged_var2] = states_var2[0]
    states4[nudged_var1] = states_var1[1]
    states4[nudged_var2] = states_var2[1]
    if np.random.random() > 0.5:
        negative_states = [tuple(states1), tuple(states4)]
        positive_states = [tuple(states2), tuple(states3)]
    else:
        negative_states = [tuple(states2), tuple(states3)]
        positive_states = [tuple(states1), tuple(states4)]
   
    return {"negative_states":negative_states, 
            "positive_states":positive_states}

if __name__ == "__main__":
    dist1d = np.array([0.1, 0.3, 0.08, 0.12, 0.03, 0.06, 0.02, 0.22, 0.07])
    dist_2vars = np.array([
        [0.1, 0.4, 0.05, 0.1],
        [0.1, 0.04, 0.15, 0.06]
    ])
    dist_3vars = np.array([
        [
            [0.01, 0.04, 0.02],
            [0.06, 0.08, 0.09]
        ],
        [
            [0.13, 0.16, 0.03],
            [0.03, 0.05, 0.02]
        ],
        [
            [0.04, 0.04, 0.05],
            [0.05, 0.09, 0.01]
        ]
    ])
    new_dist = nudge_individual(dist_3vars, 0.3)
    ProbabilityArray(new_dist)
    print(new_dist)
