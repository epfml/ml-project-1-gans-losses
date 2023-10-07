import numpy as np

def compute_mse_loss(y, tx, w):
    '''Computes the MSE loss

    This function computes the MSE loss for a linear model.

    Parameters
    ----------
    y : numpy array of shape (n, )
        The output vector of the training set
    tx : numpy array of shape (n, d)
        The input matrix of the training set (with the bias term)
    w : numpy array of shape (d, )
        The weights vector
    
    Returns
    -------
    loss : float
        The loss value
    '''
    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e ** 2)

def compute_mse_gradient(y, tx, w):
    '''Computes the MSE gradient

    This function computes the gradient of the MSE loss for a linear model.

    Parameters
    ----------
    y : numpy array of shape (n, )
        The output vector of the training set
    tx : numpy array of shape (n, d)
        The input matrix of the training set (with the bias term)
    w : numpy array of shape (d, )
        The weights vector
    
    Returns
    -------
    gradient : numpy array
        The gradient vector
    '''
    e = y - tx.dot(w)
    return -1 / len(y) * tx.T.dot(e)

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    '''Gradient Descent using MSE

    This function implements the gradient descent algorithm using the MSE loss function. We assume a linear model.

    Parameters
    ----------
    y : numpy array of shape (n, )
        The output vector of the training set
    tx : numpy array of shape (n, d)
        The input matrix of the training set (with the bias term)
    initial_w : numpy array of shape (d, )
        The initial weights vector
    max_iters : int
        The maximum number of iterations to perform
    gamma : float
        The learning rate
    
    Returns
    -------
    w : numpy array
        The final weights vector
    loss : float
        The final loss value
    '''
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_mse_gradient(y, tx, w)
        loss = compute_mse_loss(y, tx, w)
        # update w by gradient
        w = w - gamma * gradient
    return w, compute_mse_loss(y, tx, w)

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
    '''Ridge regression - L2 regularization

    This function implements ridge regression using the MSE loss function. We assume a linear model.

    Parameters
    ----------
    y : numpy array of shape (n, )
        The output vector of the training set
    tx : numpy array of shape (n, d)
        The input matrix of the training set (with the bias term)
    lambda_ : float
        The regularization parameter
    
    Returns
    -------
    w : numpy array
        The final weights vector
    loss : float
        The final loss value
    '''
    A = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.eye(np.shape(tx)[1])
    b = tx.T.dot(y)
    
    w = np.linalg.solve(A, b)

    return w, compute_mse_loss(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    raise NotImplementedError
