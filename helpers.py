"""Some helper functions for project 1."""
import csv
import numpy as np

def preprocess_data(x, y, nan_rate_threshold=0.5, in_place=False):
    """Function to preprocess the data.

    We remove the features with a nan rate higher than the threshold
    and those with no variance (std = 0) and replace the nan values
    with the mean of the feature.

    Parameters
    ----------
    x : numpy array of shape (n, d)
        The input matrix of the training set
    y : numpy array of shape (n, )
        The output vector of the training set
    nan_rate_threshold : float
        The threshold to remove features with a nan rate higher than it
    in_place : Boolean
        This variable indicates if we want to modify the input matrix or not
    
    Returns
    -------
    x : numpy array of shape (n, d')
        The input matrix of the training set after preprocessing
    y : numpy array of shape (n, )
        The output vector of the training set after preprocessing
    """
    if not in_place:
        x = x.copy()
        y = y.copy()

    # Remove features with a nan rate higher than the threshold
    nan_rate = np.sum(np.isnan(x), axis=0) / x.shape[0]
    x = x[:, nan_rate < nan_rate_threshold]

    # Replace the nan values with the mean of the feature
    nan_indices = np.where(np.isnan(x))
    x[nan_indices] = np.nanmean(x, axis=0)[nan_indices[1]]

    # Remove features with no variance
    std = np.std(x, axis=0)
    x = x[:, std != 0]

    # Normalize the data
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    return x, y



def load_csv_data(data_folder_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features)"""
    y_path = data_folder_path + "y_train.csv"
    x_path = data_folder_path + "X_train.csv"
    y = np.genfromtxt(y_path, delimiter=",", skip_header=1)
    x = np.genfromtxt(x_path, delimiter=",", skip_header=1)

    # sub-sample
    if sub_sample:
        y = y[::50]
        x = x[::50]

    return x, y


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Minibatch iterator generator

    This function generates a minibatch iterator for a dataset.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Parameters
    ----------
    y : numpy array of shape (n, )
        The output vector of the training set
    tx : numpy array of shape (n, d)
        The input matrix of the training set (with the bias term)
    batch_size : int
        the size of the minibatch
    num_batches : int
        The number of batches to generate
    shuffle : Boolean
        This variable indicates if we want to shuffle the data before generating the iterator

    Returns
    -------
    iter: iter
        The iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
