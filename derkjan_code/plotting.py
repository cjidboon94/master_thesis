import matplotlib.pyplot as plt
import numpy as np

def plot_mean_and_confidence(plot_range, mean, confidence_interval, 
                             confidence_interval_title, xlabel, ylabel, title):
    """
    Plot the mean and some kind of confidence interval (standard deviation or
    mean-squared-error)
    
    Parameters:
    ----------
    plot_range: iterable
    mean: an iterable
        the mean of the values at that point
    confidence_interval: an iterable
        Representing the  interval of confidence in that point. 
        The iterable should have length plot_range.
    confidence_interval_title:
    xlabel: label x-axis
    ylabel: label y-axis
    title: title of the plot
    
    """
    
    lower_bound = np.array(mean)-np.array(confidence_interval)
    upper_bound = np.array(mean)+np.array(confidence_interval)
    plt.plot(plot_range, mean, label="average")
    plt.fill_between(plot_range, lower_bound, upper_bound, 
                     label='{}'.format(confidence_interval_title),
                     alpha=0.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.show()
