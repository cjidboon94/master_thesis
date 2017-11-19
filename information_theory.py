import numpy as np
from scipy.stats import entropy
import probability_distributions as pb
from probability_distributions import ProbabilityArray

def calculate_mutual_information(distribution, variables1, variables2):
    """
    Calculate the mutual information between variables1 and variables2

    Parameters:
    ----------
    distribution: a probability array instance 
    variables1, variables2: set of integers

    """
    joint = distribution.marginalize(variables1.union(variables2))
    marginal1 = distribution.marginalize(variables1) 
    marginal2 = distribution.marginalize(variables2)
    return (entropy(marginal1.flatten()) + entropy(marginal2.flatten()) -
            entropy(joint.flatten()))


