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


# Implementation of SGD binary classifier
class SGD(object):
    # Constructor
    def __init__(self, eta=0.1, iters=10, seed=1):
        if eta < 0 or eta > 1.0:
            sys.exit("Eta should be between zero and one!")
        self.__eta = eta
        self.__iters = iters
        self.__seed = seed
        self.__w = []
        self.__costs = np.zeros(iters, dtype=float)

    # Learn the model
    def learn(self, x, y):
        generator = np.random.RandomState(self.__seed)
        self.__w = generator.normal(size=x.shape[1]+1)

        for i in range(self.__iters):
            r = generator.permutation(len(y))
            x, y = x[r], y[r]
            iter_cost = 0

            for xi, yi in zip(x, y):
                net_input = np.dot(xi, self.__w[1:]) + self.__w[0]
                error = yi - net_input
                self.__w[0] += self.__eta * error
                self.__w[1:] += self.__eta * xi.dot(error)

                iter_cost += 0.5 * error ** 2
                print("i= " + str(i) + " iter_cost= " + str(iter_cost) + " self_w0= " + str(self.__w[0]))
            self.__costs[i] = iter_cost / len(y)

    # Print costs figure
    def print_costs(self):
        plt.plot(range(1, len(self.__costs) + 1), self.__costs)
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.title("SGD Binary Classifier\nCosts Figure (eta = " + str(self.__eta) + " , iterations = " +
                  str(self.__iters) + " )")
        plt.grid(True)
        plt.show()
