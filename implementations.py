import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError


def mean_squared_error(y, tx, w):
    """Calculates the mean squared error"""
    e = y - tx.dot(w)
    N = len(y)

    return (1 / (2 * N)) * np.sum(e**2)


def stochastic_gradient_descent(y, tx, w):
    """Calculates the stochastic gradient descent"""
    N = len(y)
    e = y - tx.dot(w)

    return (-1 / N) * tx.T.dot(e)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Finds the best weights vector and loss for the given parameters using stochastic gradient descent"""
    w = initial_w
    loss = mean_squared_error(y, tx, w)

    for i in range(max_iters):
        grad = stochastic_gradient_descent(y, tx, w)

        w = w - gamma * grad
        loss = mean_squared_error(y, tx, w)

    return w, loss


def least_squares(y, tx):
    raise NotImplementedError


def ridge_regression(y, tx, lambda_):
    raise NotImplementedError


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    raise NotImplementedError
