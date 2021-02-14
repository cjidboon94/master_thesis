import numpy as np
import dit, random, itertools
from derkjanistic_nudges import do_derkjanistic_nudge as dj_nudge

from nudge_utils import generate_log_nudge, generate_nudge, perform_nudge, perform_log_nudge



def individual_nudge(old_X: dit.Distribution, eps: float = 0.01, rvs_other=None) -> dit.Distribution:
    mask = old_X._mask
    base = old_X.get_base()
    if old_X.outcome_length() == 1:
        return global_nudge(old_X, eps)
    outcomes = old_X.outcomes
    rv_names = old_X.get_rv_names()
    
    if rvs_other == None:
        rvs = old_X.get_rv_names()
        rvs_other = np.random.choice(rvs, len(rvs) - 1, replace=False)

    X_other, Xi_given_Xother = old_X.condition_on(rvs_other)
    nudge_size = len(Xi_given_Xother[0])

    if base == 'linear':
        nudge = generate_nudge(nudge_size, eps / len(Xi_given_Xother))
        for Xi in Xi_given_Xother:
            perform_nudge(Xi, nudge)
    else:
        nudge, sign = generate_log_nudge(nudge_size, eps)
        for Xi in Xi_given_Xother:
            perform_log_nudge(Xi, nudge, sign)
    new_X = dit.joint_from_factors(X_other, Xi_given_Xother).copy(base)
    #add back any missing outcomes
    dct = {o: new_X[o] if o in new_X.outcomes else 0.0 for o in outcomes}
    #print(outcomes, dct)
    new_X = dit.Distribution(dct) 
    new_X.set_rv_names(rv_names)
    new_X.make_dense()
    new_X._mask = mask
    return new_X


def local_nudge1(old_X: dit.Distribution, eps: float = 0.01) -> dit.Distribution:
    mask = old_X._mask
    base = old_X.get_base()
    new_X = old_X.copy(base=base)
    old_X.make_dense()
    outcomes = old_X.outcomes
    rvs = list(old_X.get_rv_names())

    random.shuffle(rvs)
    #print(rvs)
    new_Xs = np.zeros((len(rvs), len(old_X)))
    for i in range(len(rvs)):
        rvs_other = rvs[:i] + rvs[i + 1:]
        tmp = individual_nudge(old_X, eps, rvs_other=rvs_other)
        #print("tmp",tmp)
        tmp.make_dense()
        
        old_X.make_dense()
        if base == 'linear':
            new_Xs[i, :] = tmp.pmf - old_X.pmf
        else:
            new_Xs[i] = tmp.copy(base='linear').pmf - old_X.copy(base='linear').pmf
        #old_X.make_sparse()
    nudge = new_Xs.sum(axis=0)
    nudge = eps * nudge / (abs(nudge).sum())

    if base == 'linear':
        perform_nudge(new_X, nudge)
    else:
        perform_log_nudge(new_X, np.log(np.abs(nudge)), np.sign(nudge))
    new_X = dit.Distribution({o: new_X[o] if o in new_X.outcomes else 0 for o in outcomes})
    new_X.set_rv_names(rvs)
    new_X._mask = mask
    return new_X


def local_nudge2(old_X: dit.Distribution, eps: float = 0.01) -> dit.Distribution:
    mask = old_X._mask
    base = old_X.get_base()
    new_X = old_X.copy(base=base)
    old_X.make_dense()
    rvs = list(old_X.get_rv_names())

    random.shuffle(rvs)

    new_Xs = np.zeros((len(rvs), len(old_X)))
    for i in range(len(rvs)):
        rvs_other = rvs[:i] + rvs[i + 1:]
        #print(new_X.get_rv_names())
        new_X = individual_nudge(new_X, eps / len(rvs), rvs_other=rvs_other)
    return new_X


local_nudge = local_nudge2


def derkjanistic_nudge(old_X: dit.Distribution, eps: float = 0.01) -> dit.Distribution:
    base = old_X.get_base()
    outcomes = old_X.outcomes
    new_X = old_X.copy(base='linear')
    rvs = old_X.get_rv_names()
    if len(rvs) < 2:
        return global_nudge(old_X, eps)
    delta = eps / len(old_X)
    new_pmf = dj_nudge(new_X, delta)
    # print(delta)
    new_X.pmf = new_pmf
    new_X = dit.Distribution({o: new_X[o] if o in new_X.outcomes else 0 for o in outcomes})
    new_X.set_rv_names(rvs)
    new_X.normalize()
    new_X = new_X.copy(base=base)
    return new_X


def synergistic_nudge(old_X: dit.Distribution, eps: float = 0.01) -> dit.Distribution:
    base = old_X.get_base()
    outcomes = old_X.outcomes
    new_X = old_X.copy(base=base)
    rvs = old_X.get_rv_names()
    if len(rvs) < 3:
        return global_nudge(old_X, eps)

    synergy_vars = np.random.choice(len(rvs), 2, replace=False)
    states = old_X.alphabet[0]
    nudge_size = int(len(old_X) / (len(states) ** 2))
    outcome_dict = {state: np.zeros(nudge_size, dtype=int) for state in list(itertools.product(states, repeat=2))}
    for i, outcome in enumerate(old_X.outcomes):
        cur_state = outcome[synergy_vars[0]], outcome[synergy_vars[1]]
        outcome_dict[cur_state][np.argmax(outcome_dict[cur_state] == 0)] = i  # Choose the first zero entry to fill

    if base == 'linear':
        nudge = generate_nudge(nudge_size, eps / len(outcome_dict))
        perform_nudge(new_X, nudge, outcome_dict.values())
    else:
        nudge, sign = generate_log_nudge(nudge_size, eps / len(outcome_dict))
        perform_log_nudge(new_X, nudge, sign, outcome_dict.values())
    new_X = dit.Distribution({o: new_X[o] if o in new_X.outcomes else 0.0 for o in outcomes})
    new_X.set_rv_names(rvs)
    new_X.pmf[new_X.pmf == np.nan] = -np.inf
    new_X.normalize()

    return new_X


def global_nudge(old_X: dit.Distribution, eps: float = 0.01) -> dit.Distribution:
    base = old_X.get_base()
    new_X = old_X.copy(base=base)
    old_X.make_dense()
    outcomes = old_X.outcomes
    nudge_size = len(old_X)
    rvs = old_X.get_rv_names()
    if base == 'linear':
        nudge = generate_nudge(nudge_size, eps)
        perform_nudge(new_X, nudge)
    else:
        nudge, sign = generate_log_nudge(nudge_size, eps)
        perform_log_nudge(new_X, nudge, sign)
    new_X = dit.Distribution({o: new_X[o] if o in new_X.outcomes else 0 for o in outcomes})
    new_X.set_rv_names(rvs)
    return new_X