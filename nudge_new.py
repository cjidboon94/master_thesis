import numpy as np
from probability_distributions import ProbabilityArray
import probability_distributions

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
    if number_of_vars == 1:
        return nudge_states(noise_vector, dist)

    number_of_other_states = dist.flatten().shape[0]
    other_vars = np.sum(new_dist, axis=number_of_vars-1)
    other_vars = other_vars.flatten()
    cond_nudge = ProbabilityArray(dist).find_conditional(
        set([number_of_vars-1]), set(range(number_of_vars-1))
    )
    cond_nudge = np.reshape(cond_nudge, (number_of_other_states, number_of_nudge_states))

    new_dist = np.zeros((number_of_other_states, number_of_nudge_states))
    for i in range(number_of_other_states):
        nudged_states = nudge_states(noise_vector, cond_nudge[i])
        new_dist[i] = other_vars[i] * nudged_states

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
                               
    print("the positive states {}, negative states {}".format(
        positive_states, negative_states
    ))
    noise_vector[negative_states] = (
        (-noise_size/2) * np.random.dirichlet([1]*len(negative_states))
    )
    noise_vector[positive_states] = (
        (noise_size/2) * np.random.dirichlet([1]*len(positive_states))
    )
    return noise_vector

def nudge_states(noise_vector, states):
    raise NotImplementedError()

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
    #for i in range(10):
    #    print("the noise vector {} \n".format(find_noise_vector(5, 0.01)))
    

