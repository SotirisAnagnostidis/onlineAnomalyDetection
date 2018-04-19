"""
    implementation inspired from https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/
"""

import numpy as np
import random


class KPlusPlus():
    def __init__(self, number_of_centers, X):
        assert len(X) >= number_of_centers

        self.number_of_centers = number_of_centers
        self.X = X

        self.centers = []

    def _dist_from_centers(self):
        if len(self.centers) == 0:
            self.distances = np.zeros(len(self.X)) + 1
        else:
            self.distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in self.centers]) for x in self.X])

    def _choose_next_center(self):
        self.probabilities = self.distances / self.distances.sum()
        self.cumulativeProbabilities = self.probabilities.cumsum()
        r = random.random()
        ind = np.where(self.cumulativeProbabilities >= r)[0][0]
        return (self.X[ind])

    def init_centers(self):
        while len(self.centers) < self.number_of_centers:
            self._dist_from_centers()
            self.centers.append(self._choose_next_center())