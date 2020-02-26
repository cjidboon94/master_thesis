import numpy as np
import scipy.stats as st


def entropy(dist: np.ndarray):
    return st.entropy(dist, base=2)


def midpoint(min: np.ndarray, max: np.ndarray):
    mid = min * sum(max) + max * sum(min)
    return mid/sum(mid)


def sample(n_states: int, level: float):
    u = np.ones(n_states)
    d = np.random.dirichlet(u)
    max_entropy = entropy(u)
    if entropy(d)/max_entropy == level:
        return d
    else:
        support =  bin_search(d, level)
        # multiply support with a factor so that its a very narrow support
        support = 100*support
        return np.random.dirichlet(support)


def bin_search(dist: np.ndarray, level: float):
    max_entropy = entropy(np.ones(len(dist)))
    if entropy(dist) / max_entropy >= level:
        max = dist
        min = np.zeros(len(dist))
        min[np.random.randint(len(min))] = 1.
        if entropy(min) / max_entropy == level:
            return min
    else:
        max = np.ones(len(dist))
        min = dist
        if entropy(max)/max_entropy == level:
            return max

    max_ent = entropy(max)/max_entropy
    min_ent = entropy(min)/max_entropy
    while max_ent != level and min_ent != level:
        mid = midpoint(min, max)
        if entropy(mid)/max_entropy >= level:
            max = mid
            max_ent = entropy(mid)/max_entropy
        else:
            min = mid
            min_ent = entropy(mid)/max_entropy
    if entropy(max) / max_entropy == level:
        return max
    else:
        return min

