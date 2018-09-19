# ======================================================================= 
# This file is part of the CS519_Project_1 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

import sys
import os.path
import pandas as pd
import numpy as np
import sgd_bin_classifier, adaline_bin_classifier, perceptron_bin_classifier
import matplotlib.pyplot as plt


classifier = sys.argv[1]
data_path = sys.argv[2]

# Checking the classifier name
if classifier.lower() not in ["perceptron", "adaline", "sgd", "multiclass"]:
    sys.exit("Invalid classifier name!")

# Checking the file existence
if not os.path.isfile(data_path):
    sys.exit("Data file does not exist!")

# Reading the data set file
df = pd.read_csv(data_path, header=None)

# Splitting the data set in to training and testing sets with a fraction of 75%
train = df.sample(frac=0.75, random_state=1)
test = df.drop(train.index)

# Extracting the class (target) values
# We are assuming that the last column of the data set is the class
num_cols = df.shape[1]
x_tr = train.iloc[:, 0:(num_cols - 1)].values
y_tr = train.iloc[:, (num_cols - 1)].values
x_ts = test.iloc[:, 0:(num_cols - 1)].values
y_ts = test.iloc[:, (num_cols - 1)].values

# Running the classification
if classifier.lower() != "multiclass":
    # We assume the first class label as +1 and the others as -1
    positive_class = y_tr[0]
    y_tr = np.where(y_tr == positive_class, 1, -1)
    y_ts = np.where(y_ts == positive_class, 1, -1)

    # Choosing the classifier class
    if classifier.lower() == "perceptron":
        model = perceptron_bin_classifier.Perceptron(eta=0.0001, iters=20, seed=1)

    elif classifier.lower() == "adaline":
        model = adaline_bin_classifier.Adaline(eta=0.0001, iters=20, seed=1)

    elif classifier.lower() == "sgd":
        model = sgd_bin_classifier.SGD(eta=0.0001, iters=20, seed=1)

    # Learning the model
    model.learn(x_tr, y_tr)

    # Predicting the test set
    y_pred = model.predict(x_ts)

    # Evaluating the prediction (percentage of currectly classified samples)
    accuracy = ((y_ts == y_pred).sum() / y_ts.shape[0]) * 100
    print("Classifier: " + classifier.lower() + "\nAccuracy: " + str(accuracy) + "%")

    # Plot the iteration errors
    model.print_errors()
    plt.show()

else:  # Multi-class
    # A list containing the models for each class label
    models = []
    best_accuracy = 0

    # Looping through different unique values of class labels
    for label in set(y_tr):

        # Assuming the current label as +1 and the others as -1
        new_y_tr = np.where(y_tr == label, 1, -1)
        new_y_ts = np.where(y_ts == label, 1, -1)

        # Creating a new classifier for the current label and adding it to the list
        new_model = sgd_bin_classifier.SGD(eta=0.0001, iters=20, seed=1)
        models.append(new_model)

        # Learning the current model
        new_model.learn(x_tr, new_y_tr)

        # Predicting the test set using the current model
        y_pred = new_model.predict(x_ts)

        # Evaluating the new classifier and choosing the best classifier
        accuracy = ((new_y_ts == y_pred).sum() / new_y_ts.shape[0]) * 100
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_label = label

        # Plotting the iteration errors for all of the classifiers
        new_model.multi_print_errors("Positive class: " + str(label))

    print("Choosing " + str(best_label) + " as the positive class for the provided test set, yields highest accuracy"
          " which is " + str(best_accuracy) + "%")

    plt.legend()
    plt.show()
