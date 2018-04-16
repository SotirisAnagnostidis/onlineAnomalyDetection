from math import exp, log
import numpy as np
from dsio.anomaly_detectors import AnomalyMixin
import scipy.stats.distributions

def poisson(x, l):
    return_value = 1
    for x_i, l_i in zip(x, l):
        return_value *= scipy.stats.distributions.poisson.pmf(x_i, l_i)
    return return_value


class OnlineEM(AnomalyMixin):
    def __init__(self, gammas, lambdas, segment_length):
        # gammas and lambdas are the initialization
        self.gammas = np.array(gammas)
        self.lambdas = np.vstack(lambdas)
        self.segment_length = segment_length

        assert len(gammas) == len(lambdas)
        assert self.lambdas.ndim > 1

        # number of poisson mixtures
        self.m = len(gammas)
        # the dimension of the Poisson distribution
        self.dim = len(self.lambdas[0])

        # list of the gammas_i
        # each element represent the value of gamma_i for an iteration
        self.gammas_over_time = [[] for _ in gammas]
        self.lambdas_over_time = [[] for _ in lambdas]
        self.likelihood = []

        # number of current iteration
        self.iteration_k = 0

    def calculate_participation(self, data):
        """
        :param data: n array of the data to train
        :return: an (n, m) array of the participation of each data point to each poisson distribution
                m is the number of distributions
        """
        f = np.zeros(shape=(len(data), self.m))
        for i, x in enumerate(data):
            total_x = np.sum(self.gammas * np.array([poisson(x, lambda_i) for lambda_i in self.lambdas]))
            f[i] = (self.gammas * np.array([poisson(x, lambda_i) for lambda_i in self.lambdas])) / total_x
        return f

    def calculate_likelihood(self, data):
        # naive implementation for likelihood calculation
        new_likelihood = 0
        for x in data:
            total_x = np.sum(self.gammas * np.array([poisson(x, lambda_i) for lambda_i in self.lambdas]))
            new_likelihood = new_likelihood + log(total_x)
        return new_likelihood

    def update(self, data):
        """
        :param data: the batch data 
        updates gammas, lambdas and likelihood
        """

        self.iteration_k += 1
        n = len(data)
        if n <= 0:
            return
        assert len(data[0]) == len(self.lambdas[0])

        f = self.calculate_participation(data)

        # update gammas and lambdas
        temp_sum = f.sum(axis=0)

        update_factor = 1 / (pow(self.iteration_k, 0.6))

        self.gammas = (1 - update_factor) * self.gammas + update_factor * (temp_sum / n)
        
        temp = np.zeros(shape=(self.m, self.dim))
        for i, x in enumerate(data):
            temp = temp + np.vstack([x * f_i for f_i in f[i]])
        new_lambdas = np.vstack([temp[i] / temp_i for i, temp_i in enumerate(temp_sum)])

        self.lambdas = (1 - update_factor) * self.lambdas + update_factor * new_lambdas

        # append last value of gammas and lambdas
        for i, gamma_i in enumerate(self.gammas):
            self.gammas_over_time[i].append(gamma_i)

        for i, lambda_i in enumerate(self.lambdas):
            self.lambdas_over_time[i].append(lambda_i)

        self.likelihood.append(self.calculate_likelihood(data))

    def get_new_batch(self, data, pos):
        n = len(data)

        assert self.segment_length <= n

        if self.segment_length + pos <= n:
            return data[pos: pos + self.segment_length], pos + self.segment_length

        return data[pos:], n

    def train(self, data):
        """
        Runs a simulated batch processing of the whole data 
        :param data: the whole data set to train from 
        :param batch_number: the number of iterations to perform on data with barch_size self.batch_size
        """

        # the starting position of the current batch in the data
        pos = 0
        while pos < len(data):
            batch, pos = self.get_new_batch(data, pos)

            self.update(batch)

    def fit(self, x):
        """
        For fitting the initial values update function is called 
        Depending on the use of the update factor initial values may have an impact or not
        :param x: data
        """
        self.update(x)

    # TODO
    def score_anomaly(self, x):
        pass

    # TODO
    def flag_anomaly(self, x):
        pass

    def get_gammas(self):
        return self.gammas_over_time

    def get_lambdas(self):
        return self.lambdas_over_time

    # TODO average or update based on factor the final likelihood?
    def get_likelihood(self):
        return self.likelihood

    def get_bic(self, data):
        """
        :return a tuple of the bic avg_log_likelihoods and the log likelihood of the whole data
        """
        return ((-2) / self.iteration_k) * self.calculate_likelihood(data) + log(len(data)) * (2 * self.m - 1), self.calculate_likelihood(data)
