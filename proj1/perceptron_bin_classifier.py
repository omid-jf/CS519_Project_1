# ======================================================================= 
# This file is part of the CS519_Project_1 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

import numpy as np
import sys
import matplotlib.pyplot as plt


# Implementation of perceptron binary classifier
class Perceptron(object):
    # Constructor
    def __init__(self, eta=0.1, iters=10, seed=1):
        super().__init__()
        if eta < 0 or eta > 1.0:
            sys.exit("Eta should be between zero and one!")
        self.__eta = eta
        self.__iters = iters
        self.__seed = seed
        self.__w = []
        self.__errors = np.zeros(iters, dtype=int)

    # Learn the model
    def learn(self, x, y):
        generator = np.random.RandomState(self.__seed)
        self.__w = generator.normal(size=x.shape[1]+1)

        for i in range(self.__iters):
            for xi, yi in zip(x, y):
                y_hat = self.predict(xi)
                self.__errors[i] += np.where(yi == y_hat, 0, 1)

                self.__w[0] += self.__eta * (yi - y_hat)
                self.__w[1:] += self.__eta * (yi - y_hat) * xi

    # Computing the net input value
    def __net_input(self, x):
        return np.dot(x, self.__w[1:]) + self.__w[0]

    # Predict the class label of a given sample
    def predict(self, x):
        input_ = self.__net_input(x)
        return np.where(input_ >= 0, 1, -1)

    # Print errors figure
    def print_errors(self, label="", no_info=False):
        plt.plot(range(1, len(self.__errors) + 1), self.__errors, label=label)
        plt.xlabel("iteration")
        plt.ylabel("error")

        if no_info:
            plt.title("Perceptron Binary Classifier\nErrors Figure")
        else:
            plt.title("Perceptron Binary Classifier\nErrors Figure (eta = " + str(self.__eta) + " , iterations = " +
                      str(self.__iters) + " )")

        plt.grid(True)
