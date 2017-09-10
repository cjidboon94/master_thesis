import numpy as np
from probability_distributions import ProbabilityArray
import probability_distributions

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
        return entropy(old_output.flatten(), new_output.flatten(), base=2)
    else:
        raise ValueError("provide a valid measure")

def control_non_causal(dist, nudge_size, without_conditional=False):
    if not without_conditional:
        return np.reshape(
            local_non_causal(dist.flatten(), nudge_size),
            dist.shape
        )
    else:
        return np.reshape(
            local_non_causal_without_conditional(dist.flatten(), nudge_size),
            dist.shape
        )

def global_non_causal(dist, nudged_vars, nudge_size):
    """perform a multiple individual nudges

    Parameters:
    ----------
    dist: a numpy array 
        Representing a distribution, the last variable is the one
        that will be nudged
    nudged_vars: a list of ints
        Representing all the variables that will be nudged
    nudge_size: a number

    """
    old_dist = dist
    new_dist = np.copy(dist)
    total_number_of_vars = len(dist.shape)
    total_nudge = np.zeros(old_dist.shape)
    for var in nudged_vars:
        old_dist = np.swapaxes(dist, var, total_number_of_vars-1)
        new_dist = local_non_causal(old_dist, nudge_size)
        nudge = new_dist-old_dist
        old_dist = np.swapaxes(new_dist, var, total_number_of_vars-1)
        nudge = np.swapaxes(nudge, var, total_number_of_vars-1)
        total_nudge += nudge

    total_nudge_size = np.sum(np.absolute(total_nudge))
    #print("the total nudge size {}".format(total_nudge_size))
    #print("unnormalised second marginal {}".format(
    #    ProbabilityArray(dist+total_nudge).marginalize(set([1]))
    #))
    nudge_vector = total_nudge * (nudge_size/total_nudge_size)
    new_dist = np.copy(dist) + np.reshape(nudge_vector, dist.shape)
    return new_dist

def local_non_causal(dist, nudge_size):
    """Perform a local non causal nudge
    
    Parameters:
    ----------
    dist: a numpy array 
        Representing a distribution, the last variable is the one
        that will be nudged
    nudge_size: a number

    """
    number_of_vars = len(dist.shape)
    number_of_nudge_states = dist.shape[-1]
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
    cond_nudge = np.reshape(cond_nudge, (number_of_other_states, number_of_nudge_states))
    #print("other vars {}".format(other_vars))
    #print("conditional nudge {}".format(cond_nudge))

    new_dist = np.zeros((number_of_other_states, number_of_nudge_states))
    for i in range(number_of_other_states):
        nudged_states = nudge_states(noise_vector, cond_nudge[i])
        #print("the nudge_states are {}".format(nudged_states))
        new_dist[i] = other_vars[i] * nudged_states

    new_dist = np.reshape(new_dist, dist.shape)
    return new_dist

def local_non_causal_without_conditional(dist, nudge_size):
    """Perform a local non causal nudge
    
    Parameters:
    ----------
    dist: a numpy array 
        Representing a distribution, the last variable is the one
        that will be nudged
    nudge_size: a number

    """
    number_of_vars = len(dist.shape)
    number_of_nudge_states = dist.shape[-1]
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
    capped_noise[positive_states] = (
        (np.sum(capped_noise[negative_states])/np.sum(noise[negative_states]))
        * noise[positive_states]
    )
    #print("the capped noise {}".format(capped_noise))
    return probabilities+capped_noise

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
    new_dist = local_non_causal(dist_3vars, 0.3)
    ProbabilityArray(new_dist)
    print(new_dist)
