import dit
from typing import List
import numpy as np
import itertools
import scipy.stats as st

#Utility functions

def entropy(dist: np.ndarray):
    return st.entropy(dist, base=2)


def R(x, n, vs):
    x = x.reshape(-1,vs)
    for i in range(n % vs):
        x= x.T.reshape(-1,vs)
    return x


def find_max_impact(scores, input, eps):
    nudge_vector = find_minimal_subset(scores, input, eps)
    positive_nudge = sum(nudge_vector)
    nudge_vector = -nudge_vector
    nudge_vector[np.argmax(scores)] = positive_nudge
    nudge_impact = np.sum(scores*nudge_vector)
    return nudge_vector, nudge_impact


def find_minimal_subset(scores, input, eps):
    nudge_vector = np.zeros(len(input))
    sorted_indices = np.argsort(scores)

    minus_weights = np.cumsum(input[sorted_indices])
    # Find the index in the sorted indices whose weight is at most eps
    i = np.argmax(np.cumsum(minus_weights < eps))

    selected_indices = sorted_indices[:i]
    selected_weights = input[selected_indices]
    selected_weights[-1] = min(selected_weights[-1], eps - sum(selected_weights[:-1]))
    nudge_vector[selected_indices] = selected_weights
    return nudge_vector


def max_nudge(input, conditional, eps=0.01, nudge_type='individual'):
    minimal_entropy_idx = None
    if nudge_type == 'individual':
        nudge_vector, _, minimal_entropy_idx, shape = max_individual(input, conditional, eps)
        nudge_vector = nudge_vector.reshape(shape)
    elif nudge_type == 'local':
        nudge_vectors, max_impacts = max_local(input, conditional, eps)
        nudge_vector = eps * nudge_vectors.sum(axis=1) / abs(nudge_vectors).sum()

    elif nudge_type == 'synergistic':
        nudge_vector = max_synergistic(input, conditional, eps)  # nudge vector is size input
    elif nudge_type == 'global':
        nudge_vector = max_global(input, conditional, eps)  # nudge vector is size of input
    else:
        raise ValueError("type should be one of (individual, local, synergistic, global)")
    return nudge_vector, minimal_entropy_idx


def max_individual(input : dit.Distribution, conditional : List[dit.Distribution], eps: float, minimal_entropy_idx = None):
    max_impact = 0
    rvs = input.get_rv_names()
    if not minimal_entropy_idx == 0 and not minimal_entropy_idx:
        minimal_entropy_idx = np.argmin([entropy(input.marginal([rv], rv_mode='indices').pmf) for rv in len(rvs)])

    non_minimal_rvs = rvs[:minimal_entropy_idx] + rvs[minimal_entropy_idx +1:]
    non_minimal_marginal, minimal_conditional = input.condition_on(non_minimal_rvs)
    conditional = np.stack([d.pmf for d in conditional])
    conditional = conditional/conditional.sum()
    minimal_conditional = np.stack([d.pmf for d in minimal_conditional])
    indiv_shape = minimal_conditional.shape
    minimal_conditional = minimal_conditional
    nudge_vector = np.zeros()
    for allignment in itertools.product([-1, 1], repeat=conditional.shape[1]):
        allignment = np.array(allignment)
        if np.all(allignment == 1) or np.all(allignment == -1):
            continue
        scores = np.sum(allignment * conditional, axis=1)

        # Add rotation of scores so that scores are well aligned.
        scores = R(scores, minimal_entropy_idx, len(rvs))
        # Weigh scores using the non_minimal_marginal
        scores = non_minimal_marginal.pmf * scores

        vector, impact = find_max_impact(scores, minimal_conditional, eps / 2)
        if impact > max_impact:
            nudge_vector = vector
            max_impact = impact
    return nudge_vector, max_impact ,minimal_entropy_idx, indiv_shape


def max_local(input: dit.Distribution, conditional: List[dit.Distribution], eps: float):
    rvs = input.get_rv_names()
    sorted_rvs = np.argsort([entropy(input.marginal(rv).pmf) for rv in rvs])

    nudge_vectors = np.zeros(len(input), input.outcome_length()) #For each random variable we get (hopefully) a different nudge vector of len the input size
    max_impacts = np.zeros(input.outcome_length())
    for rv in sorted_rvs:
        nudge_vector, max_impacts[rv], _,_ =  max_individual(input, conditional, eps, rv)
        nudge_vectors[:, rv] = R(nudge_vector, len(sorted_rvs), len(sorted_rvs)-rv)
    return nudge_vectors, max_impacts


# TODO: Rename nudge_size impact to eps, nudge_size is the size of the vector
def max_synergistic(input: dit.Distribution, conditional: List[dit.Distribution], eps: float):
    rvs = input.get_rv_names()
    states = input.alphabet[0]
    partition_size = int(len(input) / (len(states) ** 2))
    max_entropy = (len(states) ** 2)* entropy(np.ones(partition_size))
    best_syn_vars = (0, 1)
    best_outcome_dict = {}
    lowest_entropy = max_entropy

    # conditional = np.stack([d.pmf for d in conditional]) # stack the conditional
    # conditional = conditional/conditional.sum() # normalize the conditional to give each
    for synergy_vars in itertools.combinations(range(len(rvs)), r=2):
        # Build the outcome dict
        outcome_dict = { state : np.zeros(partition_size, dtype=int) for state in list(itertools.product(states, repeat=2))}
        for i, outcome in enumerate(input.outcomes):
            cur_state = outcome[synergy_vars[0]], outcome[synergy_vars[1]]
            outcome_dict[cur_state][np.argmax(outcome_dict[cur_state] ==0)] = i #Choose the first zero entry to fill

        current_entropy = sum([entropy(input.pmf[indices]) for state, indices in outcome_dict.items()])

        if current_entropy < lowest_entropy:
            best_syn_vars = synergy_vars
            lowest_entropy = current_entropy
            best_outcome_dict = outcome_dict

    # Use best syn vars to find the nudge vector that makes the largest impact
    nudge_vector = np.zeros(len(input))

    for state, indices in best_outcome_dict.items():
        nudge_vector[indices] = max_global(input.pmf[indices], conditional[indices], eps)

    return nudge_vector, best_syn_vars


def max_global(input: dit.Distribution, conditional: List[dit.Distribution], eps: float):
    max_impact = 0
    nudge_vector = np.zeros(len(input))
    conditional = np.stack([d.pmf for d in conditional]) # stack the conditional
    conditional = conditional/conditional.sum() # normalize the conditional to give each
    for allignment in itertools.product([-1, 1], repeat=conditional.shape[1]):
        allignment = np.array(allignment)
        if np.all(allignment == 1) or np.all(allignment == -1):
            continue
        scores = np.sum(allignment*conditional, axis=1)

        vector, impact = find_max_impact(scores, input.pmf, eps / 2)
        if impact > max_impact:
            nudge_vector = vector
            max_impact = impact
    return nudge_vector


