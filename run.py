"""File for loading the dataset, training the model and making predictions."""

import numpy as np
import helpers as hlp

if __name__=="__main__":
    # Load the training data if present
    try:
        x_train, y_train = hlp.load_csv_data("data/", sub_sample=False)
    except:
        print("Training data not found. Please make sure you extracted the data.")
        exit(1)
    
    # Preprocess the data
    x_train, y_train = hlp.preprocess_data(x_train, y_train, nan_rate_threshold=0.5, in_place=True)

    #TODO: Train the model
    #TODO: Make predictions on the test set
    