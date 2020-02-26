import numpy as np
import dit

def individual(dist, eps):
    pass

def local(dist,eps):
    pass

def synergistic(dist, n_vals, eps):
    pass

def globally(dist, eps):
    if len(dist) % 2 == 0:
        nudge = np.random.permutation(np.concatenate(0.5*eps*np.random.dirichlet([1]*int(0.5*len(dist))),
                                                 -0.5*eps*np.random.dirichlet([1]*int(0.5*len(dist)))))
    else:
        u, v = int(np.floor(len(dist)/2)), int(np.ceil(len(dist)/2))
        nudge = np.random.permutation(np.concatenate(0.5*eps*np.random.dirichlet([1]*u),
                                                 -0.5*eps*np.random.dirichlet([1]*v)))
    dist += nudge
    normalize(dist)