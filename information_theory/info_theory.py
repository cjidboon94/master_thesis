import numpy as np

PERFORMANCE_MODE = False
TEST_MODE = not PERFORMANCE_MODE

def check_distribution(dist):
    if sum(dist)-1 < 1e-6:
        return True
    else:
        return False

def calculate_entropy(dist, checks_on=TEST_MODE):
    """calculate the entropy of the distribution values (a numpy array)"""
    if checks_on:
        if not check_distribution(dist):
            print("WARNING: the distribution is not normalised")

    return -np.sum(np.multiply(dist[dist != 0], np.log2(dist[dist != 0])))  
    
def calculate_mutual_information(dist1, dist2, mut_dist):
    """calculate the mutual information between the distributions
    
    params:
        dist1/dist2: A dictionary, the key is a tuple representing the
                     values within the prob. distribution and the value
                     is real number representing the probability.
        mut_dist: A dictionary. Every key is a tuple of two tuples
                  every tuple represent the values of the first/second distribion
                  while the value is a number representing the probability
    """

    return (calculate_entropy(dist1) + calculate_entropy(dist2) - 
            calculate_entropy(mut_dist))
