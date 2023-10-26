import helpers as hlp
import matplotlib.pyplot as plt
import numpy as np
from implementations import *


def convergence_plot(n_iter, gamma):
    ''' Plot the number of iterations needed to reach convergence for a given gamma value

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
    '''

    x_label = "Gamma"
    y_label = "Number of iterations"

    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Number of iterations for convergence")
    ax.plot(gamma, n_iter, 'o')
    ax.plot(gamma, n_iter, 'k-')

    plt.show()


def build_k_indices(y, k_fold, seed):
    """Builds the indices for k-fold cross-validation

    This function generates indices for k-fold cross-validation on a dataset.

    Parameters
    ----------
    y : numpy array of shape (n, )
        The target values or labels.
    k_fold : int
        The number of folds (partitions) for cross-validation.
    seed : int
        The random seed for reproducibility.

    Returns
    -------
    k_indices : numpy array of shape (k_fold, n/k_fold)
        An array of indices, where each row contains the indices for a fold.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]

    return np.array(k_indices)


def find_best_gamma_logistic(y, x, initial_w, gammas, epochs, max_iterations, threshold, threshold_loss):
    nr_iters_per_gamma = []
    for gamma in gammas:
        iters = 0
        w = initial_w.copy()
        prev_loss = 0
        last_three_diffs = 0
        for k in range(epochs):
            k_indices = build_k_indices(y, epochs, 12)
            tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
            tr_indices = tr_indices.reshape(-1)
            y_tr = y[tr_indices]
            x_tr = x[tr_indices]

            w, loss = logistic_regression(y_tr, x_tr, w, max_iterations, gamma)

            iters += 1

            if abs(prev_loss - loss) < threshold and prev_loss < threshold_loss:
                last_three_diffs += 1
                if last_three_diffs > 2:
                    nr_iters_per_gamma.append(iters)
                    break
            else:
                last_three_diffs = 0

            prev_loss = loss.copy()

        if last_three_diffs < 3:
            nr_iters_per_gamma.append(iters)

    print(nr_iters_per_gamma)

    ind_min = np.argmin(nr_iters_per_gamma)

    return gammas[ind_min]


def cross_validation_util_logistic(y, x, initial_w, k_indices, k, max_iters, gamma):
    """Perform logistic regression within a cross-validation fold and calculate loss.

    This function performs logistic regression within a specific fold of a k-fold
    cross-validation, and computes the training and testing losses.

    Parameters
    ----------
    y : numpy array of shape (n, )
        The target values or labels for the entire dataset.
    x : numpy array of shape (n, d)
        The input features for the entire dataset.
    initial_w : numpy array of shape (d, )
        The initial weights for logistic regression.
    k_indices : numpy array of shape (k_fold, n/k_fold)
        An array of indices representing the k-fold partitions of the dataset.
    k : int
        The current fold index for cross-validation.
    max_iters : int
        The maximum number of iterations for logistic regression.
    gamma : float
        The learning rate for logistic regression.

    Returns
    -------
    loss_tr : float
        The training loss for logistic regression within the current fold.
    loss_te : float
        The testing loss for logistic regression within the current fold.
    """
    te_indices = k_indices[k]
    tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indices = tr_indices.reshape(-1)
    y_te = y[te_indices]
    y_tr = y[tr_indices]
    x_te = x[te_indices]
    x_tr = x[tr_indices]

    w, loss = logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)

    return loss, compute_mse_loss(y_te, x_te, w)


def cross_validation_util_reg_logistic(y, x, initial_w, k_indices, k, max_iters, gamma, lambda_):
    """Perform regularized logistic regression within a cross-validation fold and calculate loss.

    This function performs regularized logistic regression within a specific fold of a k-fold
    cross-validation, and computes the training and testing losses.

    Parameters
    ----------
    y : numpy array of shape (n, )
        The target values or labels for the entire dataset.
    x : numpy array of shape (n, d)
        The input features for the entire dataset.
    initial_w : numpy array of shape (d, )
        The initial weights for regularized logistic regression.
    k_indices : numpy array of shape (k_fold, n/k_fold)
        An array of indices representing the k-fold partitions of the dataset.
    k : int
        The current fold index for cross-validation.
    max_iters : int
        The maximum number of iterations for logistic regression.
    gamma : float
        The learning rate for logistic regression.
    lambda_ : float
        The regularization parameter for regularized logistic regression.

    Returns
    -------
    loss_tr : float
        The training loss for regularized logistic regression within the current fold.
    loss_te : float
        The testing loss for regularized logistic regression within the current fold.
    """
    te_indices = k_indices[k]
    tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indices = tr_indices.reshape(-1)
    y_te = y[te_indices]
    y_tr = y[tr_indices]
    x_te = x[te_indices]
    x_tr = x[tr_indices]

    w, loss = reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)

    loss_tr = np.sqrt(2 * loss)
    loss_te = np.sqrt(2 * compute_mse_loss(y_te, x_te, w))

    return loss_tr, loss_te


def cross_validation_reg_logistic(y, x, k_fold, gamma, lambdas, initial_w, max_iters):
    """Perform cross-validated regularized logistic regression to find the best hyperparameters gamma and lambda.

    This function performs cross-validated regularized logistic regression on a dataset for different
    lambda values and identifies the best combination of gamma and lambda that leads to the lowest test RMSE.

    Parameters
    ----------
    y : numpy array of shape (n, )
        The target values or labels for the dataset.
    x : numpy array of shape (n, d)
        The input features for the dataset.
    k_fold : int
        The number of folds for cross-validation.
    gamma : float
        The best gamma found using logistic regression cross validation.
    lambdas : list
        A list of lambda values to be tested for regularization.
    initial_w : numpy array of shape (d, )
        The initial weights for regularized logistic regression.
    max_iters : int
        The maximum number of iterations for logistic regression.

    Returns
    -------
    best_lambda : float
        The best lambda value that minimizes test RMSE.
    best_rmse : float
        The lowest test RMSE achieved with the best gamma and lambda combination.
    """
    seed = 12
    k_fold = k_fold

    k_indices = build_k_indices(y, k_fold, seed)

    rmse_te = []

    for lambda_ in lambdas:
        rmse_te_tmp = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation_util_reg_logistic(y, x, initial_w, k_indices, k, max_iters, gamma,
                                                                  lambda_)
            rmse_te_tmp.append(loss_te)
        rmse_te.append(np.mean(rmse_te_tmp))

    ind_best = np.argmin(rmse_te)

    return lambdas[ind_best], rmse_te[ind_best]
