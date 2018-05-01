"""
    implementation inspired from https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/
"""

import numpy as np
import random
import scipy.stats.distributions


def poisson(x, l):
    return_value = 1
    for x_i, l_i in zip(x, l):
        return_value *= scipy.stats.distributions.poisson.pmf(x_i, l_i)
    if return_value == 0:
        return 
    return return_value

class KPlusPlus:
    def __init__(self, number_of_centers, x):
        assert len(x) >= number_of_centers
        assert number_of_centers > 0

        self.number_of_centers = number_of_centers
        self.x = x

        self.centers = []

    def _dist_from_centers_initialize(self):
        testing_center = self.centers[len(self.centers) - 1]
        self.distances = np.array([1/poisson(x, testing_center) for x in self.x])
        
        
    def _dist_from_centers(self):
        testing_center = self.centers[len(self.centers) - 1]
        self.distances = np.min(np.column_stack((np.array([1/poisson(x, testing_center) for x in self.x]), self.distances.T)), axis=1)

    def _choose_next_center(self):
        self.probabilities = self.distances / self.distances.sum()
        self.cumulativeProbabilities = self.probabilities.cumsum()
        r = random.random()
        ind = np.where(self.cumulativeProbabilities >= r)[0][0]
        return self.x[ind]

    def init_centers(self):
        center = random.randint(0, len(self.x))
        self.centers.append(self.x[center])
        self._dist_from_centers_initialize()
        while len(self.centers) < self.number_of_centers:
            self.centers.append(self._choose_next_center())
            if len(self.centers) < self.number_of_centers:
                self._dist_from_centers()
