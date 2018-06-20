from math import log
from dsio.anomaly_detectors import AnomalyMixin
from sklearn.cluster import KMeans
import scipy.stats.distributions
import numpy as np


def poisson(x, l):
    return_value = 1
    for x_i, l_i in zip(x, l):
        return_value *= scipy.stats.distributions.poisson.pmf(x_i, l_i)
    return return_value


def poisson_cumulative(x, l):
    return_value = 1
    for x_i, l_i in zip(x, l):
        cumulative_prob = scipy.stats.distributions.poisson.cdf(x_i, l_i)
        return_value *= min(cumulative_prob, 1 - cumulative_prob)
    return return_value


class OnlineEM(AnomalyMixin):
    def __init__(self, gammas, lambdas, segment_length, n_clusters=4, threshold='auto', update_power=1.0, verbose=0):
        """
        :param gammas: 
        :param lambdas: 
        :param segment_length: 
        :param n_clusters: the different profiles to create for the kind of users 
        :param update_power: the power that determmines the update faktor in each iteration of the online algorithm
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
        self.gammas_over_time = [[x] for x in gammas]
        self.lambdas_over_time = [[x] for x in lambdas]
        self.likelihood = []

        # number of current iteration
        self.iteration_k = 1

        self.update_power = update_power

        # a dictionary containing for each host valuable information
        self.hosts = {}

        self.n_clusters = n_clusters
        self.kMeans = KMeans(n_clusters=n_clusters, random_state=0)
        # each cluster has points in each center with a probability
        self.probabilities_per_kMean_cluster = np.zeros(shape=(n_clusters, self.m))

        # each cluster has a number of hosts in it
        self.hosts_per_kMeans_cluster = np.zeros(n_clusters)

        # each cluster has a number of points in it
        self.hard_points_per_EM_cluster = np.zeros(self.m)
        self.soft_points_per_EM_cluster = np.zeros(self.m)

        self.n_clusters = n_clusters
        if threshold == 'auto':
            self.threshold = pow(0.01, self.dim)
        else:
            self.threshold = threshold

        self.verbose = verbose

        # HMM matrix
        self.hard_transition_matrix = np.eye(self.m)
        self.soft_transition_matrix = np.eye(self.m)

    def calculate_participation(self, data):
        """
        :param data: n array of the data to train
        :return: an (n, m) array of the participation of each data point to each poisson distribution
                m is the number of distributions
        """
        f = np.zeros(shape=(len(data), self.m))
        for i, x in enumerate(data):
            participation = self.gammas * np.array([poisson(x, lambda_i) for lambda_i in self.lambdas])
            total_x = np.sum(participation)
            
            # TODO
            if total_x == 0:
                participation = np.array([1/self.m] * self.m)
                total_x = 1
            f[i] = participation / total_x
        return f

    # TODO take into account the size of the batch
    def calculate_likelihood(self, data):
        # naive implementation for likelihood calculation
        new_likelihood = 0
        for x in data:
            total_x = np.sum(self.gammas * np.array([poisson(x, lambda_i) for lambda_i in self.lambdas]))
            new_likelihood = new_likelihood + log(total_x)
        return new_likelihood

    def update_parameters(self, batch):
        """
        :param data: the batch data 
        updates gammas, lambdas and likelihood
        """

        data = batch[:, :-1]

        self.iteration_k += 1
        n = len(data)
        if n <= 0:
            return
        assert len(data[0]) == len(self.lambdas[0])

        f = self.calculate_participation(data)

        # update gammas and lambdas
        temp_sum = f.sum(axis=0)

        update_factor = 1 / (pow(self.iteration_k, self.update_power))

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

        # self.likelihood.append(self.calculate_likelihood(data))

        # upon initialization self.hosts should not contain a key for host
        # TODO memory intensive
        for point in batch:
            self.update_host(point)

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

            point_center = self.closest_centers([point])
            # point_center = np.array([-pow(x-0.5, 2) if x < 0.5 else pow(x-0.5, 2) for x in point_center]) * 2 + 0.5

            self.hosts[host]['group'] = (point_center + self.hosts[host]['group'] * host_points) / \
                                        (host_points + 1)

            # the number of data points for the host
            self.hosts[host]['n_points'] += 1

            ###
            # update transpose matrix
            previous_point = self.hosts[host]['hard_previous']

            closest_center = np.argmax(point_center)

            new_transpose = np.zeros(self.m)
            new_transpose[closest_center] = 1

            points_for_cluster = self.hard_points_per_EM_cluster[previous_point]

            self.hard_transition_matrix[previous_point] = (self.hard_transition_matrix[previous_point] *
                                                           points_for_cluster + new_transpose) / \
                                                          (points_for_cluster + 1)

            for i, previous in enumerate(self.hosts[host]['soft_previous']):
                self.soft_transition_matrix[i] = (self.soft_transition_matrix[i] * self.soft_points_per_EM_cluster[i] +
                                                  point_center * previous) / (self.soft_points_per_EM_cluster[i] +
                                                                              previous)
                self.soft_points_per_EM_cluster[i] += previous
                
                self.hosts[host]['soft_transition_matrix'] = (self.hosts[host]['soft_transition_matrix'] * self.hosts[host]['soft_points_per_cluster'][i] + point_center * previous) / (self.hosts[host]['soft_points_per_cluster'][i] +
                                                                              previous)
                self.hosts[host]['soft_points_per_cluster'][i] += previous
                

            self.hosts[host]['hard_previous'] = closest_center
            self.hosts[host]['soft_previous'] = point_center
            self.hard_points_per_EM_cluster[previous_point] += 1
            
            
            points_for_cluster_host = self.hosts[host]['points_per_cluster'][previous_point]
            self.hosts[host]['transition_matrix'][previous_point] = (self.hosts[host]['transition_matrix'][previous_point] *
                                                           points_for_cluster_host + new_transpose) / \
                                                          (points_for_cluster_host + 1)
            self.hosts[host]['points_per_cluster'][previous_point] += 1

        else:
            self.hosts[host] = {}
            # create a self.m array containing the proportion of participation for this host for every center of poisson

            point_center = self.closest_centers([point])
            self.hosts[host]['group'] = point_center

            closest_center = np.argmax(point_center)
            self.hosts[host]['hard_previous'] = closest_center
            self.hosts[host]['soft_previous'] = point_center
            # self.hosts[host]['group'] = np.array(
            #    [-pow(x - 0.5, 2) if x < 0.5 else pow(x - 0.5, 2) for x in point_center]) * 2 + 0.5

            # the number of data points for the host
            self.hosts[host]['n_points'] = 1
            
            # Host specific HMM
            self.hosts[host]['transition_matrix'] = np.eye(self.m)
            self.hosts[host]['points_per_cluster'] = np.zeros(self.m)
            self.hosts[host]['soft_transition_matrix'] = np.eye(self.m)
            self.hosts[host]['soft_points_per_cluster'] = np.zeros(self.m)

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

        pos = 0
        while pos < len(x):
            batch, pos = self.get_new_batch(x, pos)

            if self.verbose > 0:
                print('Running for data till position', pos, 'from total', len(x))

            self.update_parameters(batch)

        if self.verbose > 0:
            print('Running clustering algorithm')

        closest_centers = []

        for host in self.hosts.keys():
            closest_centers.append(self.hosts[host]['group'])

        self.kMeans.fit(closest_centers)

        for host in self.hosts.keys():
            category = self.kMeans.predict([self.hosts[host]['group']])[0]
            self.hosts[host]['category'] = category
            points_in_cluster = self.hosts_per_kMeans_cluster[category]

            self.probabilities_per_kMean_cluster[category] = \
                (self.probabilities_per_kMean_cluster[category] * points_in_cluster + self.hosts[host]['group']) / \
                (points_in_cluster + 1)

            self.hosts_per_kMeans_cluster[category] += 1

    def update(self, x):
        """
        :param data: dictionary?
        """
        # TODO (or another way to get the host name)
        #                assumes the data has the appropriate length fot batch processing

        if len(x) <= 0:
            return

        features = len(x[0])
        data = x[:, 0:features - 1]
        self.update_parameters(data)
        for point in x:
            self.update_host(point)

            # kMeans center should be updated every a number of batch updates??

    def score_anomaly_for_category(self, x, category=None, host=None):
        f = np.array([poisson_cumulative(x, lambda_i) for lambda_i in self.lambdas])
        if category is not None:
            # calculate based on the probabilities within the cluster
            gammas_for_cluster = self.probabilities_per_kMean_cluster[category]
            score_cluster = np.sum(f * gammas_for_cluster)
            if host is None:
                return score_cluster

            gammas_for_host = self.hosts[host]['group']
            score_host = np.sum(f * gammas_for_host)
            score = sum([score_cluster, score_host]) / 2
        else:
            # calculate based on the global probabilities
            score = np.sum(f * self.gammas)
        return score

    # TODO
    def score_anomaly(self, x):
        score_anomalies = np.array([])
        for point in x:
            host = point[-1]
            if host in self.hosts:
                score = self.score_anomaly_for_category(point[:-1], category=self.hosts[host]['category'], host=host)
            else:
                score = self.score_anomaly_for_category(point[:-1])

            score_anomalies = np.append(score_anomalies, [score])

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
