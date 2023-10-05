"""
Utility functions to diagnose CDE models.
"""

# 3rd party
from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np


def plot_with_uniform_band(values, ci_level, x_label, n_bins=30, figsize=(10,4), ylim=[0, 50]):
    """ Plots the PIT/HPD histogram and calculates the confidence interval for the bin values, were the PIT/HPD values
    follow an uniform distribution.

    Args:
        values: a numpy array with PIT/HPD values
        ci_level: a float between 0 and 1 indicating the size of the confidence level
        x_label: a string, populates the x_label of the plot
        n_bins: an integer, the number of bins in the histogram
        figsize: a tuple, the plot size (width, height)
        ylim: a list of two elements, including the lower and upper limit for the y axis

    Returns: The matplotlib figure object with the histogram of the PIT/HPD values and the CI for the uniform
    distribution.

    """

    # Extract the number of CDEs
    n = values.shape[0]

    # Creating upper and lower limit for selected uniform band
    ci_quantity = (1-ci_level)/2
    low_lim = binom.ppf(q=ci_quantity, n=n, p=1/n_bins)
    upp_lim = binom.ppf(q=ci_level + ci_quantity, n=n, p=1/n_bins)

    # Creating figure
    fig = plt.figure(figsize=figsize)
    plt.hist(values, bins=n_bins)
    plt.axhline(y=low_lim, color='grey')
    plt.axhline(y=upp_lim, color='grey')
    plt.axhline(y=n/n_bins, label='Uniform Average', color='red')
    plt.fill_between(x=np.linspace(0, 1, 100),
                     y1=np.repeat(low_lim, 100),
                     y2=np.repeat(upp_lim, 100),
                    color='grey', alpha=0.2)
    plt.legend(loc='best', prop={'size': 18})
    plt.xlabel(x_label, size=20)
    plt.ylim(ylim)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.close()

    return fig
