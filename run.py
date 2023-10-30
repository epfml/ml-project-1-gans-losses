"""File for loading the dataset, training the model and making predictions."""

import helpers as hlp
import numpy as np
from optimization_helpers import *
from implementations import *

if __name__ == "__main__":
    # Load the training data if present
    try:
        x_train, y_train = hlp.load_csv_data("data/", sub_sample=False)
    except:
        print("Training data not found. Please make sure you extracted the data.")
        exit(1)

    try:
        x_test = hlp.load_csv_data_test("data/")
    except:
        print("Testing data not found. Please make sure you extracted the data.")
        exit(1)

    # id will be the first id used in the submission file. It was chosen according to the example in
    # sample-submission.csv
    id = len(y_train)

    # Preprocess the data
    preprocess_config = hlp.find_preprocessing_config(x_train, categorical_threshold=3)
    x_train = hlp.preprocess_data_config(
        x_train, preprocess_config, nan_rate_threshold=0.43, in_place=True
    )

    x_train, y_train = hlp.balance_data(x_train, y_train)

    x_test = hlp.preprocess_data_config(
        x_test, preprocess_config, nan_rate_threshold=0.43, in_place=True
    )

    # initial_w will be a vector containing normal distributed values between 0 and 1
    initial_w = np.random.normal(0.5, 0.2, x_train.shape[1])
    initial_w = np.clip(initial_w, 0, 1)

    max_iters = 600
    gamma = 0.3

    # The value of lambda_ was chosen experimentally using the function cross_validation_reg_logistic
    lambda_ = 0.0004175

    w, loss = reg_logistic_regression(
        y_train, x_train, lambda_, initial_w, max_iters, gamma
    )

    # We compute the y predictions using our trained model. Each value will be approximated to -1 or 1,
    # according to what the AICrowd submission is expecting.
    y_pred = x_test.dot(w)
    y_pred = np.where(y_pred < 0.5, -1, np.where(y_pred >= 0.5, 1, y_pred))

    ids = [id + i for i in range(len(y_pred))]

    hlp.create_csv_submission(ids, y_pred, "y_test.csv")
