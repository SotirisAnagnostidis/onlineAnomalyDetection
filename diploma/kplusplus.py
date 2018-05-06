"""
    implementation inspired from https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/
"""

import numpy as np
import random
import scipy.stats.distributions
import sys


def poisson(x, l):
    return_value = 1
    for x_i, l_i in zip(x, l):
        return_value *= scipy.stats.distributions.poisson.pmf(x_i, l_i)
    if return_value == 0:
        return sys.float_info.epsilon
    return return_value

class KPlusPlus:
    def __init__(self, number_of_centers, x, stochastic=False, stochastic_n_samples=10000, random_seed=42):
        """
        :param stochastic: When stochastic is True for faster calculation only keep a smaller subset 
                            of the data of size stochastic_n_samples
        """
        assert len(x) >= number_of_centers
        assert number_of_centers > 0

        self.number_of_centers = number_of_centers
        if stochastic and stochastic_n_samples < len(x):
            idx = np.random.randint(len(x), size=stochastic_n_samples)
            self.x = x[idx,:]
        else:
            self.x = x

        self.overflow_avoid = len(x) + 1
        self.centers = []
        self.random_seed = random_seed
        
    def _distances(self, center):
        # the maximum mass probability value is for the center itself
        # this is definitely an integer as the centers are chosen from the dataaset
        max_value = poisson(center, center)
        return np.array([1/poisson(x, center) - 1/max_value for x in self.x])

    def _dist_from_centers_initialize(self):
        testing_center = self.centers[len(self.centers) - 1]
        self.distances = self._distances(testing_center)
        
    def _dist_from_centers(self):
        testing_center = self.centers[len(self.centers) - 1]
        self.distances = np.min(np.column_stack((self._distances(testing_center), self.distances.T)), axis=1)

    def _choose_next_center(self):
        # avoid overflow
        self.distances[self.distances > np.finfo(np.float64).max / self.overflow_avoid] =  np.finfo(np.float64).max / self.overflow_avoid
        
        self.probabilities = self.distances / self.distances.sum()
        self.cumulativeProbabilities = self.probabilities.cumsum()
        r = random.random()
        ind = np.where(self.cumulativeProbabilities >= r)[0][0]
        return self.x[ind]

    def init_centers(self, verbose=0):
        random.seed(self.random_seed)
        
        center = random.randint(0, len(self.x))
        self.centers.append(self.x[center])
        if verbose > 0:
            print('Centers found:', len(self.centers))
        self._dist_from_centers_initialize()
        while len(self.centers) < self.number_of_centers:
            self.centers.append(self._choose_next_center())
            if verbose > 0:
                print('Centers found:', len(self.centers))
            if len(self.centers) < self.number_of_centers:
                self._dist_from_centers()
