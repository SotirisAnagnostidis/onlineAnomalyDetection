"""
    implementation inspired from https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/
"""

import numpy as np
import random


class KPlusPlus:
    def __init__(self, number_of_centers, x):
        assert len(x) >= number_of_centers

        self.number_of_centers = number_of_centers
        self.x = x

        self.centers = []

    def _dist_from_centers(self):
        testing_center = self.centers[len(self.centers) - 1]
        if len(self.centers) == 1:
            self.distances = np.array([np.linalg.norm(x - testing_center) ** 2 for x in self.x])
        else:
            self.distances = np.min(np.column_stack((np.array([np.linalg.norm(x - testing_center) ** 2 for x in self.x]), self.distances.T)), axis=1)

    def _choose_next_center(self):
        self.probabilities = self.distances / self.distances.sum()
        self.cumulativeProbabilities = self.probabilities.cumsum()
        r = random.random()
        ind = np.where(self.cumulativeProbabilities >= r)[0][0]
        return self.x[ind]

    def init_centers(self):
        center = random.randint(0, len(self.x))
        self.centers.append(self.x[center])
        while len(self.centers) < self.number_of_centers:
            self._dist_from_centers()
            self.centers.append(self._choose_next_center())
