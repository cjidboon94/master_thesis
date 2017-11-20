import json
import os
import numpy as np
from scipy import stats

def get_folder_name(path, input_or_conditional, distribution_type, number_of_variables, number_of_states=5):
    """
    Parameters:
    ----------
    path: a string, the path to the code folder of the master thesis
    input_or_conditional: a string, either "input" or "conditional"
    distribution_type: a string, 
        in the set: {"dirichlet", "entropy75", "entropy50", "max_distance"}
    number_of_variables: an int between 1 and 6
    number_of_states: an int (for now only 5 is supported)
    
    """
    if input_or_conditional == "input":
        dist_folder = "input_distributions/"
    elif input_or_conditional == "conditional":
        dist_folder = "conditional_output_distributions/"
        
    if distribution_type == "dirichlet":
        specific_dist_folder = "dirichlet"
    elif distribution_type == "entropy75" and input_or_conditional == "input":
        specific_dist_folder = "entropy_0.75"
    elif distribution_type == "entropy50" and input_or_conditional == "input":
        specific_dist_folder = "entropy_0.5"
    elif distribution_type == "max_distance" and input_or_conditional == "conditional":
        specific_dist_folder = "distance_optimization"
    else:
        print("provide a valid distribution type")
        raise ValueError()
        
    file_folder = "{}var_{}states".format(number_of_variables, number_of_states)
    return path + dist_folder + specific_dist_folder + '/' + file_folder + '/'

def generate_distributions(path_to_files, file_format, distribution_numbers):
    for i in distribution_numbers:
        file_name = path_to_files + file_format.format(i)
        #print(file_name)
        with open(file_name, 'rb') as f:
            yield np.load(f)

def get_input_dists(distribution_path, distribution_type, distribution_numbers, number_of_vars, number_of_states=5):
    dist_folder = get_folder_name(
        distribution_path, "input", distribution_type, number_of_vars, number_of_states
    )
    return generate_distributions(dist_folder, "dist_{}.npy", distribution_numbers)
    
def get_conditional_output_dists(distribution_path, distribution_type, distribution_numbers, 
                                 number_of_vars, number_of_states=5):
    dist_folder = get_folder_name(
        distribution_path, "conditional", distribution_type, number_of_vars, number_of_states
    )
    return generate_distributions(dist_folder, "cond_dist_{}.npy", distribution_numbers)     

if __name__ == "__main__":
    DISTRIBUTION_PATH = "/home/joboti/azumi_derkjan/master_thesis/code/"
    dists = get_input_dists(DISTRIBUTION_PATH, 'entropy50', [0, 1, 2, 3, 4], 3)
    for dist in dists:
        print(stats.entropy(dist.flatten()))

    dists = get_conditional_output_dists(DISTRIBUTION_PATH, 'max_distance', [0, 1, 2, 3, 4], 3)
    for dist in dists:
        print(np.sum(dist.flatten()))
