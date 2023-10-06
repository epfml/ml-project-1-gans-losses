import numpy as np

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError


def mean_squared_error(y, tx, w):
    e = y - tx.dot(w)
    N = len(y)

    return (1 / (2 * N)) * np.sum(e ** 2)

def stochastic_gradient_descent(y, tx, w):
    N = len(y)
    e = y - tx.dot(w)

    return (-1 / N) * tx.dot(e)

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for i in range(max_iters):
        loss = mean_squared_error(y, tx, w)
        grad = stochastic_gradient_descent(y, tx, w)

        w = w - gamma * grad

    return w, loss

def least_squares(y, tx):
    raise NotImplementedError

def ridge_regression(y, tx, lambda_):
    raise NotImplementedError

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    raise NotImplementedError
