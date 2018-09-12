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
                y_hat = self.__predict(xi)
                self.__errors[i] += np.where(yi == y_hat, 0, 1)

                self.__w[0] += self.__eta * (yi - y_hat)
                self.__w[1:] += self.__eta * (yi - y_hat) * xi

    # Predict the class label for a given sample
    def __predict(self, x):
        input_ = np.dot(self.__w[1:], x) + self.__w[0]
        v = np.where(input_ >= 0, 1, -1)
        return v

    # Print errors figure
    def print_errors(self):
        plt.plot(range(1, len(self.__errors) + 1), self.__errors)
        plt.xlabel("iteration")
        plt.ylabel("error")
        plt.title("Perceptron Binary Classifier\nErrors Figure (eta = " + str(self.__eta) + " , iterations = " +
                  str(self.__iters) + " )")
        plt.grid(True)
        plt.show()
