import numpy as np
import pandas as pd

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

    dist_nonzero = dist[dist!=0]
    return -np.sum(np.multiply(dist_nonzero, np.log2(dist_nonzero)))  
    
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

def points_to_dist(points):
    sel_points_df = pd.DataFrame(points)
    a = pd.DataFrame(sel_points_df.groupby(list(sel_points_df.columns)).size())
    b = a.values.flatten()
    return b/np.sum(b) 
