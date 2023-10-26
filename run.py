"""File for loading the dataset, training the model and making predictions."""

import numpy as np
import helpers as hlp

from implementations import *
from cross_validation import *

if __name__ == "__main__":
    # Load the training data if present
    try:
        x, y = hlp.load_csv_data("data/", sub_sample=False)
    except:
        print("Training data not found. Please make sure you extracted the data.")
        exit(1)

    # Preprocess the data
    x, y = hlp.preprocess_data(x, y, nan_rate_threshold=0.5, in_place=True)

    # Split data to have 85% for training and 15% for testing to get accuracy
    split_point = int(0.85 * x.shape[0])

    x_train = x[:split_point]
    y_train = y[:split_point]

    x_test = x[split_point:]
    y_test = y[split_point:]

    gammas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    lambdas = np.logspace(-10, -2, 10)

    initial_w = np.random.normal(0.5, 0.2, x.shape[1])
    initial_w = np.clip(initial_w, 0, 1)

    k_fold = 8

    max_iterations = 100
    epochs = 100
    threshold = 0.001

    threshold_loss = 0.67

    best_gamma = find_best_gamma_logistic(y, x, initial_w, gammas, epochs, max_iterations, threshold, threshold_loss)

    print(best_gamma)

    # best_lambda, best_rmse = cross_validation_reg_logistic(y, x, k_fold, best_gamma, lambdas, initial_w,
    #                                                                    max_iterations)
    #
    # print(best_lambda, best_rmse)
