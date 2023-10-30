# Heart Attack Prediction

This project, conducted as part of the Machine Learning Course at EPFL, focuses on using machine learning to predict heart attacks based on health-related data from over 300,000 individuals. 
We imposed data preprocessing, including the removal of irrelevant columns, standardization, and one-hot encoding for categorical data. We also address class imbalance through data balancing. 
We chose Logistic Regression and Regularized Logistic Regression as our models, with careful hyperparameter tuning. 
This project highlights the importance of data preparation and model selection in healthcare analytics. In the following sections I 
will briefly describe the most important functions used for data preprocessing, hyperparameter tuning and the regressions themselves.
For more information about each function we recommend reading the function documentation provided at the beginning of the functions.

### Data Preprocessing

In the following section I will briefly describe the functions used for data preprocessing:
- find_preprocessing_config(x, categorical_threshold) : This function creates a preprocessing configuration that is used 
by the function *_preprocess_data_config_*. It computes the mean, the standard deviation and the nan rate of each feature.
- preprocess_data_config(x, config, nan_rate_threshold, in_place) : The function is used to preprocess the data based on 
the preprocessing configuration. It removes the features with a nan rate higher than the threshold and those 
with no variance, standardizes the data and replaces the remaining nan values with 0.
- balance_data(x, y, in_place) : This function is used to balance the rate of each unique value in y.

All the functions in this section are implemented in *_helpers.py_*.

### HyperParameter Tuning

This section will briefly describe the function used for choosing the best hyperparameters:

- find_best_gamma(y, x, initial_w, gammas, max_iterations) : The function is used to find the fastest converging gamma parameter for logistic regression using an iterative approach.
- cross_validation_reg_logistic(y, x, k_fold, gamma, lambdas, initial_w, max_iters) : This function performs cross-validated regularized logistic regression to find the best hyperparameter lambda.

The functions in this section are implemented in *_optimization_helpers.py_*.

### Logistic and Regularized Logistic Regressions

In this section we will present the functions used for logistic regression and regularized logistic regression:

- logistic_regression(y, tx, initial_w, max_iters, gamma) : This function performs logistic regression using gradient descent optimization.
- reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma) : The function performs regularized logistic regression using gradient descent optimization.

The functions in this section are implemented in *_implementations.py_*.

## Training and Running the model
To train and run the model, a user should call *python _run.py_*. For this to work, the x_train.csv, y_train.csv, x_test.csv
files should be inside a folder called data, which is located in the same folder as the script *_run.py_*. The script generates
a y_test.csv file, which contains the results of running on x_test.csv the model trained on x_train.csv and y_train.csv using Regularized 
Logistic Regression with gamma 0.3 and lambda 0.0004175(found using cross-validation) for a number of 600 iterations.