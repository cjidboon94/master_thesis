import scipy.stats as st
import numpy as np



def entropy(dist: np.ndarray):
    return st.entropy(dist, base=2)

def rel_entropy(dist: np.ndarray):
    uniform = np.ones(dist.size)
    return st.entropy(dist,base=2)/st.entropy(uniform,base=2)

def KL(p: np.ndarray, q: np.ndarray):
    if len(p) != len(q):
        return np.inf
    return st.entropy(p,q,base=2)