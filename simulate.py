import collections
import numpy as np

def calculate_batch_std(samples_ord_dict, batch_size):
    """
    Calculate the standard deviation of the batches
    
    Parameters:
    ----------
    samples_ord_dict: an OrderedDict
    batch_size: an integer 
    
    Returns:
    -------
    A list of numbers. The first entry in the list represent the 
    standard deviation of the batches of of the first values in 
    the OrderedDict.
    
    """
    batches_std = []
    for measure, samples in samples_ord_dict.items():
        batched_estimates = []
        for i in range(len(samples)/batch_size):
            batched_estimates.append(
                np.mean(samples[i*batch_size:(i+1)*batch_size])
            )

        batches_std.append(np.std(batched_estimates))

    return batches_std

def find_mean_std_mse(samples_dict, batch_size=None):
    """
    For experiment find for every value the mean, the standard deviation,
    and the standard deviation of the batches
    
    Parameters:
    ----------
    samples_dict: a dict
        Every key in the dict represent some quantity of a variable
        of interest, i.e. the number of variables. The value should be
        an iterable of samples gathered
    batch_size: integer
        Should be a divisor of the number of samples in samples dict
        
    Returns:plot_range, mean, standard_deviation, batch_MSE
    -------
    plot_range: iterable
        An iterable containing the values for which the experiment 
        was performed
    mean: iterable
        The mean value for every value the experiment was performed
    standard_deviation: iterable
        The standard deviation for every value of the experiment
    batch_std: None, if batch_size is None, otherwise iterable 
        represent the standard deviation for the batches
        
    """
    sorted_ordered_samples_dict = collections.OrderedDict(
        sorted(samples_dict.items(), key=lambda x: x[0])
    )
    plot_range = sorted_ordered_samples_dict.keys()
    mean = [np.mean(samples) for samples 
            in sorted_ordered_samples_dict.values()]
    standard_deviation = [np.std(samples) for samples 
                          in sorted_ordered_samples_dict.values()]

    if batch_size is None:
        return plot_range, mean, standard_deviation, None
    else:
        if len(list(samples_dict.values())[0])%batch_size != 0:
            raise ValueError("batch size should be a divisor of the number of samples")
            
        batches_std = calculate_batch_std(sorted_ordered_samples_dict, batch_size)
        
    return plot_range, mean, standard_deviation, batches_std
