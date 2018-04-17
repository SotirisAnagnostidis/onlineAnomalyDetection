from math import log
import numpy as np
from dsio.anomaly_detectors import AnomalyMixin
from dsio.update_formulae import decision_rule
from sklearn.cluster import KMeans
import scipy.stats.distributions



def poisson(x, l):
    return_value = 1
    for x_i, l_i in zip(x, l):
        return_value *= scipy.stats.distributions.poisson.pmf(x_i, l_i)
    return return_value


class OnlineEM(AnomalyMixin):
    def __init__(self, gammas, lambdas, segment_length, n_clusters=4, threshold=0.0001):
        """
        :param gammas: 
        :param lambdas: 
        :param segment_length: 
        :param n_clusters: the different profiles to create for the kind of users 
        """
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

        # a dictionary containing for each host valuable information
        self.hosts = {}

        self.n_clusters = n_clusters
        self.kMeans = KMeans(n_clusters=n_clusters, random_state=0)
        # each cluster has points in each center with a probability
        self.probabilities_per_kMean_cluster = np.zeros(shape=(n_clusters, self.m))
        # each cluster has a number of hosts in it
        self.counts_per_kMeans_cluster = np.zeros(n_clusters)

        self.threshold = threshold

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

    def update_parameters(self, data):
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

    def closest_centers(self, data):
        n = len(data)

        f = self.calculate_participation(data)

        # update gammas and lambdas
        temp_sum = f.sum(axis=0)
        return temp_sum / n

    def update_host(self, point):
        host = point[-1]
        if host in self.hosts:
            host_points = self.hosts[host]['n_points']

            self.hosts[host]['group'] = (self.closest_centers([point]) + self.hosts[host]['group'] * host_points) / \
                                        (host_points + 1)

            # the number of data points for the host
            self.hosts[host]['n_points'] += 1
        else:
            self.hosts[host] = {}
            # create a self.m array containing the proportion of participation for this host for every center of poisson
            self.hosts[host]['group'] = self.closest_centers([point])

            # the number of data points for the host
            self.hosts[host]['n_points'] = 1

    def fit(self, x):
        """
        For fitting the initial values update function is called the pth column holds the by attribute
        x is a array n times p where 
        :param x: data
        """
        if len(x) <= 0:
            return

        features = len(x[0])
        # the starting position of the current batch in the data
        data = x[:, 0:features - 1]
        pos = 0
        while pos < len(data):
            batch, pos = self.get_new_batch(data, pos)

            self.update_parameters(batch)

        # upon initialization self.hosts should not contain a key for host
        for point in x:
            self.update_host(point)

        closest_centers = []

        for host in self.hosts.keys():
            closest_centers.append(self.hosts[host]['group'])

        self.kMeans.fit(closest_centers)

        for host in self.hosts.keys():
            category = self.kMeans.predict([self.hosts[host]['group']])[0]
            self.hosts[host]['category'] = category
            points_in_cluster = self.counts_per_kMeans_cluster[category]

            self.probabilities_per_kMean_cluster[category] = \
                (self.probabilities_per_kMean_cluster[category] * points_in_cluster + self.hosts[host]['group']) / \
                (points_in_cluster + 1)

            self.counts_per_kMeans_cluster[category] += 1

    def update(self, x):
        """
        :param data: dictionary?
        """
        # TODO (or another way to get the host name)
        #                assumes the data has the appropriate length fot batch processing

        data = x[:, 0:features - 1]
        self.update_parameters(data)
        for point in x:
            self.update_host(point)

        # kMeans center should be updated every a number of batch updates??

    # TODO
    def score_anomaly(self, x):
        score_anomalies = np.array([])
        for point in x:
            host = point[-1]
            f = self.gammas * np.array([poisson(point[:-1], lambda_i) for lambda_i in self.lambdas])
            if host in self.hosts:
                # calculate based on the probabilities within the cluster
                gammas_for_cluster = self.probabilities_per_kMean_cluster[self.hosts[host]['category']]
                decision = np.max(f * gammas_for_cluster)
            else:
                # calculate based on the global probabilities
                decision = np.max(f * self.gammas)

            score_anomalies = np.append(score_anomalies, [decision])

        return score_anomalies

    # TODO
    def flag_anomaly(self, x):
        return self.score_anomaly(x) < self.threshold

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
        return ((-2) / self.iteration_k) * self.calculate_likelihood(data) + log(len(data)) * (
        2 * self.m - 1), self.calculate_likelihood(data)
    