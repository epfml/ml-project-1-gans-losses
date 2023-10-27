"""Some helper functions for project 1."""
import csv
import numpy as np

def balance_data(x, y, in_place=False):
    ''' Function to balance the rate of each unique value in y

    Parameters
    ----------
    x : numpy array of shape (n, d)
        The input matrix of the training set
    y : numpy array of shape (n, )
        The output vector of the training set
    
    Returns
    -------
    x_balanced : numpy array of shape (n, d)
        The input matrix of the training set balanced
    y_balanced : numpy array of shape (n, )
        The output vector of the training set balanced
    '''
    if not in_place:
        x = x.copy()
        y = y.copy()

    unique, counts = np.unique(y, return_counts=True)

    # Get the number of samples for the most frequent class
    max_count = np.max(counts)

    # Get the indices of the samples for each class
    indices = {}
    for i in range(len(unique)):
        indices[unique[i]] = np.where(y == unique[i])[0]
    
    # Balance each class by randomly sampling from it
    for i in range(len(unique)):
        samples_to_append = max_count - counts[i]
        indices_to_append = np.random.choice(indices[unique[i]], samples_to_append)
        x = np.append(x, x[indices_to_append], axis=0)
        y = np.append(y, y[indices_to_append], axis=0)
    
    return x, y

def find_preprocessing_config(x, categorical_threshold=3):
    """Function to find the preprocessing configuration.

    We need the preprocessing configuration to preprocess the test set.

    We compute the mean, the standard deviation and the nan rate of each feature.

    Parameters
    ----------
    x : numpy array of shape (n, d)
        The input matrix of the training set
    categorical_threshold : int
        The threshold to determine if a feature is categorical or not,
        namely if the number of unique values of the feature is lower than it
    
    Returns
    -------
    config: dict from column index to necessary preprocessing information
        The preprocessing information for one column consists of:
            - nan_rate: the nan rate of the column
            - categorical: a boolean indicating if the column is categorical or not
            - categories: the unique values of the column if it is categorical
            - std: the standard deviation of the column
            - mean: the mean of the column
    """

    config = {}
    nan_rate = np.sum(np.isnan(x), axis=0) / x.shape[0]
    std = np.nanstd(x, axis=0)
    mean = np.nanmean(x, axis=0)
    categorical = np.array([len(np.unique(x[:, i])) < categorical_threshold for i in range(x.shape[1])])
    for i in range(x.shape[1]):
        config[i] = {"nan_rate": nan_rate[i],
                     "std": std[i],
                     "mean": mean[i]}
        unique_values = np.unique(x[:, i])
        unique_values = unique_values[~np.isnan(unique_values)]
        config[i]["categorical"] = len(unique_values) < categorical_threshold
        if config[i]["categorical"]:
            config[i]["categories"] = unique_values

    return config


def preprocess_data_config(x, config, nan_rate_threshold=0.5, in_place=False):
    """Function to preprocess the data based on the preprocessing configuration.

    We remove the features with a nan rate higher than the threshold
    and those with no variance (std = 0). We standardieze the data
    and replace the remaining nan values with 0 (as it is the mean
    of the standardized data).

    Parameters
    ----------
    x : numpy array of shape (n, d)
        The input matrix of the training set
    config: dict from column index to necessary preprocessing information
        The preprocessing information for one column consists of:
            - nan_rate: the nan rate of the column
            - std: the standard deviation of the column
            - mean: the mean of the column
    nan_rate_threshold : float (between 0 and 1)
        The threshold to remove features with a nan rate higher than it
    in_place : Boolean
        This variable indicates if we want to modify the input matrix or not
    """

    if not in_place:
        x = x.copy()

    # Figure out which features to remove
    to_remove = []
    valid_features = lambda x, to_remove: set(range(x.shape[1])) - set(to_remove)
    for i in range(x.shape[1]):
        if config[i]["nan_rate"] > nan_rate_threshold:
            to_remove.append(i)
        elif config[i]["std"] == 0:
            to_remove.append(i)

    for i in valid_features(x, to_remove):
        if not config[i]["categorical"]:
            # Standardize the non-categorical data
            x[:, i] = (x[:, i] - config[i]["mean"]) / config[i]["std"]

            # Replace the nan values with 0
            nan_indices = np.where(np.isnan(x[:, i]))
            x[nan_indices, i] = 0
        else:
            # One-hot encoding the categorical data
            categories = config[i]["categories"]
            is_category_column = np.zeros_like(x[:, i])
            for value in categories:
                new_column = (x[:, i] == value).astype(int)
                is_category_column = np.logical_or(is_category_column, new_column)
                x = np.column_stack((x, new_column))

            # One-hot encoding the values different than the categories
            x = np.column_stack((x, is_category_column))

            to_remove.append(i)

    # Remove the columns that we don't need anymore
    x = np.delete(x, to_remove, axis=1)

    return x


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


def load_csv_data_test(test_folder_path):
    """Loads data and returns tX (features)"""
    x_path = test_folder_path + "x_test.csv"
    x = np.genfromtxt(x_path, delimiter=",", skip_header=1)

    return x


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w", newline='') as csvfile:
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
