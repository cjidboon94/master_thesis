import numpy as np
import scipy.stats as st
from math import isclose


def entropy(dist: np.ndarray):
    return st.entropy(dist, base=2)


def midpoint(min: np.ndarray, max: np.ndarray):
    mid = min * sum(max) + max * sum(min)
    return mid/sum(mid)


def sample(n_states: int, level: float):
    if level < 0 or level >1:
        raise ValueError("level should be between 0 and 1")
    u = np.ones(n_states)
    d = np.random.dirichlet(u)
    max_entropy = entropy(u)
    if entropy(d)/max_entropy == level:
        return d
    else:
        support, sample_again =  bin_search(d, level)
        # multiply support with a factor so that its a very narrow support
        if sample_again:
            support = 1000*support
            return np.random.dirichlet(support)
        else:
            return support

def bin_search(dist: np.ndarray, level: float):
    max_entropy = entropy(np.ones(len(dist)))
    if entropy(dist) / max_entropy >= level:
        max = dist
        min = np.zeros(len(dist))
        min[np.random.randint(len(min))] = 1.
        if entropy(min) / max_entropy == level:
            return min, False
    else:
        max = np.ones(len(dist))
        min = dist
        if entropy(max)/max_entropy == level:
            return max, False

    max_ent = entropy(max)/max_entropy
    min_ent = entropy(min)/max_entropy
    while not isclose(max_ent, level) and not isclose(min_ent, level):
        mid = midpoint(min, max)
        if entropy(mid)/max_entropy >= level:
            max = mid
            max_ent = entropy(mid)/max_entropy
        else:
            min = mid
            min_ent = entropy(mid)/max_entropy
    if isclose(entropy(max) / max_entropy, level):
        return max, True
    else:
        return min, True