"""File for loading the dataset, training the model and making predictions."""

import numpy as np
import helpers as hlp

if __name__=="__main__":
    # Load the training data if present
    try:
        x, y = hlp.load_csv_data("data/", sub_sample=False)
    except:
        print("Training data not found. Please make sure you extracted the data.")
        exit(1)
    
    # Preprocess the data
    preprocess_config = hlp.find_preprocessing_config(x, categorical_threshold=3)
    x = hlp.preprocess_data_config(x, preprocess_config, nan_rate_threshold=0.5, in_place=True)

    # Split data to have 85% for training and 15% for testing to get accuracy
    split_point = int(0.85*x.shape[0])

    x_train = x[:split_point, :]
    y_train = y[:split_point]

    x_test = x[split_point:, :]
    y_test = y[split_point:]

    #TODO: Train the model
    #TODO: Make predictions on the test set

    # Result of passing x_train through the model, remove when predition done
    y_result = np.array([])

    accuracy = y_result[np.equal(y_test, y_result)].shape[0] / y_test.shape[0]
    print(accuracy)




    