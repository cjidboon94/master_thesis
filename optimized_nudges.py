import dit
from typing import List
import numpy as np
import itertools
import scipy.stats as st
from nudge_utils import generate_log_nudge, generate_nudge, perform_nudge, perform_log_nudge
from derkjanistic_nudges import max_derkjanistic_nudge as max_dj_nudge
from evo_configs import get_config
from measures import entropy

#Utility functions

# Utility functions
def R(x, n, vs, states=3):
    x = x.reshape(-1, states)
    for i in range(n % vs):
        x = x.T.reshape(-1, states)
    return x


def find_max_impact(scores, input, eps):
    nudge_vector = find_minimal_subset(scores, input, eps)
    # print("in find max impact, nudge vector found:", nudge_vector)
    positive_nudge = sum(nudge_vector)
    nudge_vector = -nudge_vector
    nudge_vector[np.argmax(scores)] = positive_nudge
    nudge_impact = np.sum(scores * nudge_vector)

    return nudge_vector, nudge_impact


def find_minimal_subset(scores, input, eps):
    nudge_vector = np.zeros(len(input))
    sorted_indices = np.argsort(scores)
    minus_weights = np.cumsum(input[sorted_indices])
    # Find the index in the sorted indices whose weight is at most eps
    i = np.argmax(np.cumsum(minus_weights < eps))

    selected_indices = sorted_indices[:i]
    # print(selected_indices)
    # print("the indices",selected_indices)
    if len(selected_indices) > 0:
        selected_weights = input[selected_indices]
        # print(selected_weights, selected_weights.shape)
        selected_weights[-1] = np.amin([selected_weights[-1], eps - sum(selected_weights[:-1])])
    else:
        selected_indices = sorted_indices[0]
        selected_weights = np.amin([eps, abs(eps - input[selected_indices])])
    nudge_vector[selected_indices] = selected_weights
    # print(nudge_vector, input, eps)
    return nudge_vector


def max_nudge(input, conditional, eps=0.01, nudge_type='individual', minimal_entropy_idx=None):
    minimal_entropy_idx = None
    if nudge_type == 'individual':
        nudge_vector, _, minimal_entropy_idx = max_individual(input, conditional, eps, minimal_entropy_idx)
    elif nudge_type == 'local':
        nudge_vectors, max_impacts = max_local(input, conditional, eps)
        nudge_vector = nudge_vectors

    elif nudge_type == 'synergistic_old':
        nudge_vector, _ = max_synergistic(input, conditional, eps)  # nudge vector is size input
    elif nudge_type == 'derkjanistic':
        nudge_vector, _ = max_derkjanistic(input, conditional, eps)
    elif nudge_type == 'global':
        nudge_vector = max_global(input, conditional, eps)  # nudge vector is size of input
    else:
        raise ValueError("type should be one of (individual, local, synergistic, global)")
    return nudge_vector, minimal_entropy_idx


def max_individual(input: dit.Distribution, conditional: np.ndarray, eps: float = 0.01,
                   minimal_entropy_idx=None):
    rvs = input.get_rv_names()
    conditional = conditional / conditional.sum()
    states = len(input.alphabet[0])
    if not minimal_entropy_idx == 0 and not minimal_entropy_idx:
        minimal_entropy_idx = np.argmin(
            [entropy(input.marginal([rv], rv_mode='indices').pmf) for rv in range(len(rvs))])

    non_minimal_rvs = rvs[:minimal_entropy_idx] + rvs[minimal_entropy_idx + 1:]
    non_minimal_marginal, minimal_conditional = input.condition_on(non_minimal_rvs)
    [d.make_dense() for d in minimal_conditional]
    # minimal_conditional = np.stack([d.pmf for d in minimal_conditional])
    # print("minimal_conditional:",minimal_conditional)
    indiv_shape = (len(minimal_conditional), len(minimal_conditional[0]))

    # minimal_conditional = minimal_conditional.flatten()
    nudge_vector = np.zeros(indiv_shape)
    rotated_conditional = R(conditional, minimal_entropy_idx, len(rvs), states)
    total_max_impact = 0
    # print(len(rvs), (eps / 2)/len(minimal_conditional))
    for i, mc_dist in enumerate(minimal_conditional):

        rows = rotated_conditional[i * states:(i + 1) * states, :]

        max_impact = 0
        for allignment in itertools.product([-1, 1], repeat=rotated_conditional.shape[1]):
            allignment = np.array(allignment)
            if np.all(allignment == 1) or np.all(allignment == -1):
                continue
            scores = np.sum(allignment * rows, axis=1)

            # Add rotation of scores so that scores are well aligned.
            # Weigh scores using the non_minimal_marginal

            vector, impact = find_max_impact(scores, mc_dist.pmf, (eps / 2) / len(minimal_conditional))
            if impact > max_impact:
                nudge_vector[i, :] = vector
                max_impact = impact
        total_max_impact += max_impact
    return nudge_vector, total_max_impact, minimal_entropy_idx


def max_local(input: dit.Distribution, conditional: np.ndarray, eps: float = 0.01):
    rvs = input.get_rv_names()
    sorted_rvs = np.argsort([entropy(input.marginal([rv], rv_mode='indices').pmf) for rv in range(len(rvs))])
    nudge_vectors = np.zeros((input.outcome_length(), int(len(input) / 3),
                              3))  # For each random variable we get (hopefully) a different nudge vector of len the input size
    max_impacts = np.zeros(input.outcome_length())
    for rv in sorted_rvs:
        nudge_vectors[rv, :, :], max_impacts[rv], _ = max_individual(input, conditional, eps / len(sorted_rvs), rv)
    return nudge_vectors, max_impacts


def max_synergistic(input: dit.Distribution, conditional: np.ndarray, eps: float = 0.01):
    rvs = input.get_rv_names()
    states = input.alphabet[0]
    partition_size = int(len(input) / (len(states) ** 2))
    max_entropy = (len(states) ** 2) * entropy(np.ones(partition_size))
    best_syn_vars = (0, 1)
    best_outcome_dict = {}
    lowest_entropy = max_entropy

    # conditional = np.stack([d.pmf for d in conditional]) # stack the conditional
    # conditional = conditional/conditional.sum() # normalize the conditional to give each
    for synergy_vars in itertools.combinations(range(len(rvs)), r=2):
        # Build the outcome dict
        outcome_dict = {state: np.zeros(partition_size, dtype=int) for state in
                        list(itertools.product(states, repeat=2))}
        for i, outcome in enumerate(input.outcomes):
            cur_state = outcome[synergy_vars[0]], outcome[synergy_vars[1]]
            outcome_dict[cur_state][np.argmax(outcome_dict[cur_state] == 0)] = i  # Choose the first zero entry to fill

        current_entropy = sum([entropy(input.pmf[indices]) for state, indices in outcome_dict.items()])

        if current_entropy < lowest_entropy:
            best_syn_vars = synergy_vars
            lowest_entropy = current_entropy
            best_outcome_dict = outcome_dict

    # Use best syn vars to find the nudge vector that makes the largest impact
    nudge_vector = np.zeros(len(input))

    for state, indices in best_outcome_dict.items():
        nudge_vector[indices] = max_global(input.pmf[indices],
                                           np.array([d for i, d in enumerate(conditional) if i in indices]),
                                           eps / len(best_outcome_dict), False)

    return nudge_vector, best_syn_vars


def max_derkjanistic(input: dit.Distribution, conditional: np.ndarray,
                     eps: float = 0.01) -> dit.Distribution:
    rvs = input.get_rv_names()
    outcomes = input.outcomes
    evo_params = get_config(len(rvs))
    max_dj_nudge_found = max_dj_nudge(input, conditional, eps, evo_params)
    new_X = max_dj_nudge_found.new_dist 
    
    dct = {o: new_X[o] if o in new_X.outcomes else 0.0 for o in outcomes}
    #print(outcomes, dct)
    new_X = dit.Distribution(dct) 
    new_X.set_rv_names(rvs)
    return new_X


def max_global(input: dit.Distribution, conditional: np.ndarray, eps: float = 0.01, first=True):
    max_impact = 0
    nudge_vector = np.zeros(len(input))
    #conditional = np.stack([d.copy('linear').pmf for d in conditional])  # stack the conditional
    conditional = conditional / conditional.sum()  # normalize the conditional to give each
    for allignment in itertools.product([-1, 1], repeat=conditional.shape[1]):
        allignment = np.array(allignment)
        if np.all(allignment == 1) or np.all(allignment == -1):
            continue
        alligned_conditional = allignment * conditional
        # print("a",allignment, allignment.shape)
        # print("c",conditional, conditional.shape)
        # print("ac",alligned_conditional, alligned_conditional.shape)
        scores = np.sum(alligned_conditional, axis=1)
        if first:
            vector, impact = find_max_impact(scores, input.pmf, eps / 2)
        else:
            vector, impact = find_max_impact(scores, input, eps / 2)
        # print("impacts", impact, max_impact)
        if impact > max_impact:
            nudge_vector = vector
            max_impact = impact
    # print(nudge_vector, sum(abs(nudge_vector)))
    return nudge_vector


def max_individual_nudge(old_X: dit.Distribution, YgivenX: np.ndarray, eps: float = 0.01):
    if old_X.outcome_length() == 1:
        return max_global_nudge(old_X, YgivenX, eps)
    nudges, minimal_idx = max_nudge(old_X.copy('linear'), YgivenX, eps=eps, nudge_type='individual')
    # print("individual eps",sum([sum(abs(nudge)) for nudge in nudges]), eps, old_X.outcome_length())
    return do_max_individual_nudge(old_X, nudges, minimal_idx)


def do_max_individual_nudge(old_X, nudges, minimal_idx, from_local=False):
    mask = old_X._mask
    base = old_X.get_base()
    rvs = old_X.get_rv_names()
    outcomes = old_X.outcomes
    states = len(old_X.alphabet[0])
    
    non_minimal_rvs = rvs[:minimal_idx] + rvs[minimal_idx + 1:]
    Xother, Xi_given_Xother = old_X.condition_on(non_minimal_rvs)
    old_shape = len(old_X)
    old_Xis = [Xi.copy() for Xi in Xi_given_Xother]
    for nudge, Xi in zip(nudges, Xi_given_Xother):
        # if from_local:
        #    print("before_nudge", nudge, Xi)

        if base == 'linear':
            perform_nudge(Xi, nudge)
        else:
            log_nudge, sign = np.log(np.abs(nudge)), np.sign(nudge)
            perform_log_nudge(Xi, log_nudge, sign)
        # if from_local:
        #   print("after_nudge",Xi)
    new_X = dit.joint_from_factors(Xother, Xi_given_Xother).copy(base)
    new_X.make_dense()
    new_shape = len(new_X)
    x = new_X.pmf.reshape(-1, states)
    y = [np.all(r == -np.inf) for r in x]
    row_deleted = np.any(y)

    if from_local and new_shape != old_shape or from_local and row_deleted:
        print("nudges:", nudges)
        print("old_X:", old_X.pmf.reshape(-1, 3))
        print("new_X", new_X.pmf.reshape(-1, 3))
        print("old Xis", np.vstack([oXi.pmf for oXi in old_Xis]))
        print("new Xis", np.vstack([nXi.pmf for nXi in Xi_given_Xother]))
    
    dct = {o: new_X[o] if o in new_X.outcomes else 0.0 for o in outcomes}
    #print(outcomes, dct)
    new_X = dit.Distribution(dct) 
    new_X.set_rv_names(rvs)
    new_X._mask = mask
    return new_X


def max_local_nudge1(old_X: dit.Distribution, YgivenX: np.ndarray, eps: float = 0.01):
    if old_X.outcome_length() == 1:
        return max_global_nudge(old_X, YgivenX, eps)

    mask = old_X._mask
    base = old_X.get_base()
    new_X = old_X.copy(base=base)
    old_X.make_dense()
    outcomes = old_X.outcomes
    rvs = old_X.get_rv_names()
    
    individual_nudges, _ = max_nudge(old_X.copy('linear'), YgivenX, eps=eps, nudge_type='local')
    new_Xs = np.zeros((old_X.outcome_length(), len(old_X)))
    for i, nudges in enumerate(individual_nudges):
        tmp = do_max_individual_nudge(old_X, nudges, i)

        # print(i, tmp.pmf, old_X.pmf, new_Xs)
        new_Xs[i, :] = tmp.pmf - old_X.pmf

    nudge = new_Xs.sum(axis=0)
    nudge = eps * nudge / (abs(nudge).sum())

    if base == 'linear':
        perform_nudge(new_X, nudge)
    else:
        log_nudge, sign = np.log(np.abs(nudge)), np.sign(nudge)
        perform_log_nudge(new_X, log_nudge, sign)
    dct = {o: new_X[o] if o in new_X.outcomes else 0.0 for o in outcomes}
    #print(outcomes, dct)
    new_X = dit.Distribution(dct) 
    new_X.set_rv_names(rvs)
    new_X._mask = mask
    return new_X


def max_local_nudge2(old_X: dit.Distribution, YgivenX: np.ndarray, eps: float = 0.01):
    if old_X.outcome_length() == 1:
        return max_global_nudge(old_X, YgivenX, eps)

    mask = old_X._mask
    base = old_X.get_base()
    new_X = old_X.copy(base=base)
    old_X.make_dense()
    rvs = old_X.get_rv_names()
    sorted_rvs = np.argsort([entropy(old_X.marginal([rv], rv_mode='indices').pmf) for rv in range(len(rvs))])
    oldshape = len(old_X)
    outcomes = old_X.outcomes
    # print("before", new_X.pmf.shape)
    for i, rv in enumerate(sorted_rvs):
        nudges, _ = max_nudge(new_X.copy('linear'), YgivenX, eps=(eps / len(sorted_rvs)),
                              nudge_type='individual', minimal_entropy_idx=rv)
        #        print("local eps",sum([sum(abs(nudge)) for nudge in nudges]), eps, old_X.outcome_length())
        new_X = do_max_individual_nudge(new_X, nudges, rv, True)
        # print("after {}".format(i), new_X.pmf.shape)
        new_X.make_dense()
        newshape = len(new_X)
    #  if oldshape != newshape:
    #      print(nudges)
    #   print("after {} and making dense".format(i), new_X.pmf.shape)
    dct = {o: new_X[o] if o in new_X.outcomes else 0.0 for o in outcomes}
    #print(outcomes, dct)
    new_X = dit.Distribution(dct) 
    new_X.set_rv_names(rvs)
    new_X._mask = mask
    return new_X


max_local_nudge = max_local_nudge2


def max_synergistic_nudge(old_X: dit.Distribution, YgivenX: np.ndarray, eps: float = 0.01):
    base = old_X.get_base()
    new_X = old_X.copy(base=base)
    old_X.make_dense()
    rvs = old_X.get_rv_names()
    outcomes = old_X.outcomes
    if len(rvs) < 3:
        return max_global_nudge(old_X, YgivenX, eps)

    nudge, _ = max_nudge(old_X.copy('linear'), YgivenX, eps=eps, nudge_type='synergistic_old')
    #  print("synergistic eps",sum(abs(nudge)), eps, old_X.outcome_length())
    if base == 'linear':
        perform_nudge(new_X, nudge)
    else:
        log_nudge, sign = np.log(np.abs(nudge)), np.sign(nudge)
        perform_log_nudge(new_X, log_nudge, sign)
    dct = {o: new_X[o] if o in new_X.outcomes else 0.0 for o in outcomes}
    #print(outcomes, dct)
    new_X = dit.Distribution(dct) 
    new_X.set_rv_names(rvs)
    return new_X


def max_derkjanistic_nudge(old_X: dit.Distribution, YgivenX: np.ndarray, eps: float = 0.01):
    rvs = old_X.get_rv_names()
    if len(rvs) < 2:
        return max_global_nudge(old_X, YgivenX, eps)

    base = old_X.get_base()
    
    new_X = max_derkjanistic(old_X.copy('linear'), YgivenX, eps)
    return new_X.copy(base)


def max_global_nudge(old_X: dit.Distribution, YgivenX: np.ndarray, eps: float = 0.01):
    base = old_X.get_base()
    new_X = old_X.copy(base=base)
    old_X.make_dense()
    rvs = old_X.get_rv_names()
    outcomes = old_X.outcomes
    
    nudge, _ = max_nudge(old_X.copy('linear'), YgivenX, eps=eps, nudge_type='global')

    #  print("global eps",sum(abs(nudge)), eps, old_X.outcome_length())
    if base == 'linear':
        perform_nudge(new_X, nudge)
    else:
        # print(nudge)
        log_nudge, sign = np.log(np.abs(nudge)), np.sign(nudge)
        # print(log_nudge, sign)
        # log_nudge[log_nudge == -np.inf] = 0
        # print("converted to log nudge",nudge, log_nudge, sign)
        perform_log_nudge(new_X, log_nudge, sign)
    
    dct = {o: new_X[o] if o in new_X.outcomes else 0.0 for o in outcomes}
    #print(outcomes, dct)
    new_X = dit.Distribution(dct) 
    new_X.set_rv_names(rvs)
    return new_X



