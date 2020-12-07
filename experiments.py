import numpy as np
from dist_utils import generate_distribution, get_marginals
import dit
from dit.validate import InvalidNormalization
from dit.exceptions import  ditException

from nudges import individual_nudge, local_nudge, synergistic_nudge, derkjanistic_nudge, global_nudge
from optimized_nudges import max_individual_nudge, max_local_nudge, max_synergistic_nudge, max_derkjanistic_nudge, max_global_nudge

def experiment(inputs):
    level, n_vars, dists, interventions, seed = inputs
    np.random.seed(seed)
    n_states = 3

    nudges = [individual_nudge, local_nudge, synergistic_nudge, derkjanistic_nudge, global_nudge]
    # nudges = [individual_nudge, local_nudge, synergistic_nudge, global_nudge]
    means = np.zeros((dists, len(nudges)))
    for i in range(dists):
        XY = generate_distribution(n_vars + 1, n_states, level)  # +1 for the output variable
        old_Y = XY.marginal('Y').copy('linear')
        old_X, YgivenX = get_marginals(XY)
        [Y.make_dense() for Y in YgivenX]
        oldest_X = old_X.copy()

        intervention_results = np.zeros((len(nudges), interventions))
        for j in range(interventions):
            for idx, nudge in enumerate(nudges):
                if not np.allclose(old_X.pmf, oldest_X.pmf):
                    raise ValueError("Something went wrong during {}. Original X has changed".format(nudge.__name__))
                new_X = None
                try:
                    new_X = nudge(old_X)
                except IndexError as e:
                    print(level, n_vars, nudge, e)
                    raise e
                try:
                    new_Y = dit.joint_from_factors(new_X, YgivenX).marginal('Y').copy('linear')
                except InvalidNormalization as e:
                    print(nudge)
                    print('new_x = {}'.format(sum(new_X.pmf)), new_X.pmf)
                    raise e
                except ditException as e:
                    print(nudge)
                    old_X.make_dense()
                    new_X.make_dense()
                    print(level, n_vars, seed, i, old_X.pmf, new_X.pmf)
                    print(level, n_vars, seed, i,
                          "oldX has {} outcomes, newX has {} outcomes\nYgivenX has {} cond distributions,old_x has 0 outcomes at{}".format(
                              len(old_X), len(new_X), len(YgivenX), np.flatnonzero(old_X.pmf == 0)))
                    old_X.make_sparse()
                    raise e
                new_Y.make_dense()
                intervention_results[idx, j] = sum(
                    abs(new_Y.pmf - old_Y.pmf))  # np.linalg.norm(new_Y.pmf - old_Y.pmf, ord=1)
        means[i, :] = np.median(intervention_results, axis=1)
    print(level, n_vars, "done")
    return (level, n_vars), means


def optim_experiment(inputs):
    level, n_vars, dists, interventions, seed = inputs
    np.random.seed(seed)
    n_states = 3

    nudges = [max_individual_nudge, max_local_nudge, max_synergistic_nudge,
              max_derkjanistic_nudge, max_global_nudge]
    means = np.zeros((dists, len(nudges)))
    for i in range(dists):
        XY = generate_distribution(n_vars + 1, n_states, level)  # +1 for the output variable
        old_Y = XY.marginal('Y').copy('linear')
        old_X, YgivenX = get_marginals(XY)
        # print(YgivenX[0].get_base())
        [Y.make_dense() for Y in YgivenX]
        oldest_X = old_X.copy()
        # print("YgX old", [Y.pmf for Y in YgivenX])

        intervention_results = np.zeros((len(nudges), interventions))
        for j in range(interventions):
            for idx, nudge in enumerate(nudges):

                new_X = nudge(old_X, YgivenX)
                try:
                    new_XY = dit.joint_from_factors(new_X, YgivenX)
                    new_Y = new_XY.marginal('Y').copy('linear')
                except dit.exceptions.ditException as e:
                    print(level, n_vars, nudge, e)
                    raise e
                new_Y.make_dense()
                # print(idx, "X", old_X.copy('linear').pmf, new_X.copy('linear').pmf)
                # print(idx, "XY", XY.copy('linear').pmf, new_XY.copy('linear').pmf)
                # print(idx, "Y", old_Y.pmf, new_Y.pmf)
                intervention_results[idx, j] = sum(abs(new_Y.pmf - old_Y.pmf))

        means[i, :] = np.median(intervention_results, axis=1)

    print(level, n_vars, "done")
    return (level, n_vars), means