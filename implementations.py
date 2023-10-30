import numpy as np
from helpers import batch_iter


def compute_mse_loss(y, tx, w):
    """Computes the MSE loss

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
    """
    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
    """Computes the MSE gradient

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
    """
    e = y - tx.dot(w)
    return -1 / len(y) * tx.T.dot(e)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient Descent using MSE

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
    """
    w = initial_w
    for _ in range(max_iters):
        # compute gradient and loss
        gradient = compute_mse_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma * gradient
    return w, compute_mse_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient Descent using MSE

    This function implements the stochastic gradient descent algorithm using the MSE loss function
    and a batch size of 1. We assume a linear model.

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
    """

    w = initial_w
    loss = compute_mse_loss(y, tx, w)

    for _ in range(max_iters):
        for mini_y, mini_tx in batch_iter(y, tx, 1):
            grad = compute_mse_gradient(mini_y, mini_tx, w)

            w = w - gamma * grad
            loss = compute_mse_loss(y, tx, w)

    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations

    This function implements the least square algorithm using the MSE loss function.

    Parameters
    ----------
    y : numpy array of shape (n, )
        The output vector of the training set
    tx : numpy array of shape (n,d)
        The input matrix of the training set (with the bias term)

    Returns
    -------
    w : numpy array
        The final weights vector
    loss : float
        The final loss value
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    return w, compute_mse_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression - L2 regularization

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
    """
    A = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.eye(np.shape(tx)[1])
    b = tx.T.dot(y)

    w = np.linalg.solve(A, b)

    return w, compute_mse_loss(y, tx, w)


def sigmoid(t):
    """Sigmoid Activation Function

    This function implements the sigmoid activation function.

    Parameters
    ----------
    t : numpy array or scalar
        The input to the sigmoid function.

    Returns
    -------
    numpy array or scalar
        The output of the sigmoid function, which is in the range [0, 1].
    """
    return 1 / (1 + np.exp(-t))


def calculate_log_loss(y, tx, w):
    """Negative Log-Likelihood Loss Function

    This function calculates the negative log-likelihood loss for logistic regression.

    Parameters
    ----------
    y : numpy array of shape (n, )
        The output vector of the training set
    tx : numpy array of shape (n, d)
        The input matrix of the training set (with the bias term)
    w : numpy array of shape (d, )
        The weights vector.

    Returns
    -------
    float
        The negative log-likelihood loss value.
    """
    return np.sum(np.log(1 + np.exp(tx.dot(w))) - y * tx.dot(w)) / len(y)


def calculate_log_grad(y, tx, w):
    """Calculate the gradient for Logistic Regression

    This function computes the gradient of the negative log-likelihood loss function.

    Parameters
    ----------
    y : numpy array of shape (n, )
        The output vector of the training set
    tx : numpy array of shape (n, d)
        The input matrix of the training set (with the bias term)
    w : numpy array of shape (d, )
        The weights vector.

    Returns
    -------
    numpy array of shape (d, )
        The gradient vector.

    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y) / len(y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic Regression

     This function performs logistic regression using gradient descent optimization.

     Parameters
     ----------
    y : numpy array of shape (n, )
         The output vector of the training set
     tx : numpy array of shape (n, d)
         The input matrix of the training set (with the bias term)
     initial_w : numpy array of shape (d, )
         The initial weights vector.
     max_iters : int
         The maximum number of iterations for gradient descent.
     gamma : float
         The learning rate.

     Returns
     -------
     w : numpy array
         The final weights vector after optimization.
     loss : float
         The final loss value after optimization.

    """
    w = initial_w

    for _ in range(max_iters):
        grad = calculate_log_grad(y, tx, w)

        w = w - gamma * grad
    loss = calculate_log_loss(y, tx, w)

    return w, loss


def calculate_reg_grad(y, tx, w, lambda_):
    """Calculate the gradient for Logistic Regression

    This function computes the gradient of the negative log-likelihood loss function.

    Parameters
    ----------
    y : numpy array of shape (n, )
        The output vector of the training set
    tx : numpy array of shape (n, d)
        The input matrix of the training set (with the bias term)
    w : numpy array of shape (d, )
        The weights vector.
    lambda_ : float
        The regularization parameter

    Returns
    -------
    numpy array of shape (d, )
        The gradient vector.

    """
    grad = tx.T.dot(sigmoid(tx.dot(w)) - y) / y.size
    return grad + 2 * lambda_ * w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized Logistic Regression

     This function performs regularized logistic regression using gradient descent optimization.

     Parameters
     ----------
    y : numpy array of shape (n, )
        The output vector of the training set
    tx : numpy array of shape (n, d)
        The input matrix of the training set (with the bias term)
    lambda_ : float
        The regularization parameter
    initial_w : numpy array of shape (d, )
        The initial weights vector.
    max_iters : int
        The maximum number of iterations for gradient descent.
    gamma : float
        The learning rate.

    Returns
    -------
    w : numpy array
         The final weights vector after optimization.
    loss : float
         The final loss value after optimization.

    """
    w = initial_w

    for _ in range(max_iters):
        grad = calculate_reg_grad(y, tx, w, lambda_)

        w = w - gamma * grad

    loss = calculate_log_loss(y, tx, w)
    return w, loss
