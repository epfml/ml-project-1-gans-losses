import matplotlib.pyplot as plt
import numpy as np


def convergence_plot(n_iter, gamma):
    """Plot the number of iterations needed to reach convergence for a given gamma value

    Parameters
    ----------
    n_iter : list of int
        The number of iterations needed to reach convergence for each gamma value
    gamma : list of float
        The gamma values for which the number of iterations is computed

    Returns
    -------
    None

    Shows the final plot
    """

    x_label = "Gamma"
    y_label = "Number of iterations"

    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Number of iterations for convergence")
    ax.plot(gamma, n_iter, "o")
    ax.plot(gamma, n_iter, "k-")
    plt.show()


def nan_rate_ccdf(x):
    """Plot the complementary cumulative distribution function of the nan rate

    Parameters
    ----------
    x : numpy array of shape (n, d)
        The input matrix of the training set

    Returns
    -------
    None

    Shows the final plot
    """

    x_label = "Nan rate"
    y_label = "Probability"

    nan_rate = np.sum(np.isnan(x), axis=0) / x.shape[0]
    nan_rate = np.sort(nan_rate)
    nan_rate = nan_rate[::-1]
    cdf = np.arange(1, nan_rate.shape[0] + 1) / nan_rate.shape[0]

    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("P(Nan rate > x)")
    ax.plot(nan_rate, cdf, "o")
    ax.plot(nan_rate, cdf, "k-")
    plt.show()


def plot_errors(lambdas, errors):
    """Plot the errors for different lambdas

    Parameters
    ----------
    lambdas : numpy array of shape (n, )
        The lambdas used for the ridge regression
    errors : numpy array of shape (n, )
        The errors for each lambda

    Returns
    -------
    None

    Shows the final plot
    """

    x_label = "Lambda"
    y_label = "Error"

    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Error for different lambdas")
    ax.plot(lambdas, errors, "o")
    ax.plot(lambdas, errors, "k-")
    plt.show()
