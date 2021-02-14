import distribution_sampler as sampler
import numpy as np
import dit, string, itertools
from typing import List
import pickle
import os

def get_vars(n_vars: int) -> List[str]:
    vs = ['X{}'.format(i) for i in range(n_vars-1)]
    vs.append('Y')
    return vs

def get_labels(n_vars: int, n_states: int) -> List[str]:
    if n_states < 1 or n_states > 10:
        raise ValueError("states should be greater than 0 and  less than or equal to 10")
    return [''.join(i) for i in itertools.product(string.digits[:n_states], repeat=n_vars)]


def generate_distribution(n_vars: int, n_states: int, entropy_level: float, base=np.e) -> dit.Distribution:
    var_names = get_vars(n_vars)
    state_labels = get_labels(n_vars, n_states)
    pmf = sampler.sample(n_states**n_vars, level=entropy_level)
    if base == np.e:
        pmf = np.log(pmf)
    d = dit.Distribution(state_labels, pmf=pmf, base=base)
    d.set_rv_names(var_names)
    return d

def get_marginals(d: dit.Distribution) -> (dit.Distribution, List[dit.Distribution]):
    rvs = d.get_rv_names()[:-1]  #everything except the output
    return d.condition_on(rvs)

def get_joint(X: dit.Distribution, YgX: List[dit.Distribution], base=np.e) -> dit.Distribution:
    return dit.joint_from_factors(X, YgX).copy(base=base)

def print_conditional(YgX):
    for i, Y in enumerate(YgX):
        Y.make_dense()
        print("{}: ".format(i), Y.pmf)
        
def load_dist(model, parameter, n_vars,distribution_number):
    if model == "ising":
        folder = "ising_distributions"
        p_folder = "temp{:.2f}".format(parameter)
        
    if model == "sis":
        folder = "sis_distributions"
        beta, gamma = parameter
        p_folder = "beta{:.2f}".format(beta)
        distribution_number = str(distribution_number) + "_gamma{:.2f}".format(gamma)
        
    dist_file = "{}/{}/n{}/d{}.pkl".format(folder,p_folder,n_vars, distribution_number)
    if os.path.exists(dist_file):
        with open(dist_file, "rb") as f:
            dist = pickle.load(f)
        return dist
    else:
        return None

def calculate_XY(X, YgX, model="ising"):
    X.make_dense()
    a = X.alphabet
    outcomes = list(itertools.product(a, repeat=X.outcome_length()+1))
    XY_pmf = X.pmf[:,np.newaxis]*YgX
    XYd = {o:p for o, p in zip(outcomes,XY_pmf)}
    XY = dit.ScalarDistribution(XYd)
    return XY
    
def calculate_Y(X, YgX, model="ising"):
    X.make_dense()
    Y_pmf = (X.pmf[:,np.newaxis]*YgX).sum(axis=0)
    outcomes = [-1,1] if model == "ising" else [0,1]
    Yd = {o:p for o, p in zip(outcomes,Y_pmf)}
    Y = dit.ScalarDistribution(Yd)
    return Y