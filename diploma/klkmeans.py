from numpy.linalg import norm
import random as ran
import numpy as np
from math import log
from scipy.linalg import eig 

def find_stationary(transition_matrix):
    S, U = eig(transition_matrix.T)
    stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
    stationary = stationary / np.sum(stationary)
    return np.abs(stationary)


def kl_distance(P, Q):
    total_distance = 0
    for i in range(len(P)):
        if Q[i] == 0 or P[i] == 0:
            continue
        total_distance += P[i] * log(P[i]/Q[i])
            
    return total_distance

def kl_distance2(P, Q):
    total_distance = 0
    for i in range(len(P)):
        if P[i] == 0:
            continue
        total_distance += P[i] * log(P[i]/((Q[i] + P[i])/2))
            
    return total_distance

def solveStationary(A):
    """ x = xA where x is the answer
    x - xA = 0
    x( I - A ) = 0 and sum(x) = 1
    """
    n = A.shape[0]
    a = np.eye( n ) - A
    a = np.vstack( (a.T, np.ones( n )) )
    b = np.matrix( [0] * n + [ 1 ] ).T
    return np.squeeze(np.asarray(np.linalg.lstsq( a, b )[0]))

def distance(A1, A2):
    stationary = solveStationary(A1)
    return sum([stationary[i] * (2*kl_distance2(A1[i], A2[i])) for i in range(len(stationary))])
    #return sum([(2*kl_distance2(A1[i], A2[i])) for i in range(len(stationary))])

def hmm_distance(A1, A2):
    return (distance(A1, A2) + distance(A2, A1)) / 2

class kMeans:
    def __init__(self, em, n_clusters=3, initial_centers=None, n_iters=20, n_runs=10):
        self.em = em
        self.initial_centers = initial_centers
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.n_runs = n_runs
        
    def _transition_matrices_for_cluster(self, k, assignments):
        members = [key for key, value in assignments.items() if value == k]

        matrices = []
        for host in members:
            matrices.append(self.em.hosts[host]['transition_matrix'])

        return np.array(matrices)

    def _compute_centroids(self, assignments):
        C = np.zeros(shape=(self.n_clusters, self.em.m, self.em.m), dtype='d')
        
        for k in range(self.n_clusters):

            if not (np.array(list(assignments.values())) == k).any():
                continue
                
            matrices = self._transition_matrices_for_cluster(k, assignments)
            C[k] = np.mean(matrices, axis=0)
        return C

    def _cost(self, C, assignments):
        cost = 0
        for k in range(self.n_clusters):
            matrices = self._transition_matrices_for_cluster(k, assignments)
            for transition_matrix in matrices:
                cost += hmm_distance(transition_matrix, C[k])
        return cost
        
    def run(self):
        min_cost = float('+inf')
        best_C = None
        best_assignment = None
        
        for _ in range(self.n_runs):
            print('Starting run')
            # random initialize the assignment of each host to a cluster
            assignments = dict(zip(self.em.hosts, np.random.randint(0, self.n_clusters, len(self.em.hosts))))
            
            C = self._compute_centroids(assignments)
            
            C, assignments = self._base_kmeans(C)
            clust_cost = self._cost(C, assignments)

            if clust_cost < min_cost:
                min_cost = clust_cost
                best_C = C.copy()
                best_assignment = assignments.copy()
            
        self.assignments = best_assignment
        self.centers = best_C   
        return best_C, best_assignment


    def _base_kmeans(self, C):
        n = len(self.em.hosts)

        C_final = C

        #KMeans algorithm
        cent_dists = None
        assignments = None
        prev_assignments = None
        best_shift = None

        iters = self.n_iters
        converged = False

        while iters != 0 and not converged:
            #assign elements to new clusters    
            assignments = {}
            for host in self.em.hosts:
                distances = np.array([hmm_distance(self.em.hosts[host]['transition_matrix'], C_final[i]) 
                                      for i in range(self.n_clusters)])
                assignments[host] = np.argmin(distances)

            #check if converged, if not compute new centroids
            if prev_assignments is not None and prev_assignments == assignments:
                converged = True
                print('converged')
            else: 
                C_final = self._compute_centroids(assignments)
                print('The cost is', self._cost(C_final, assignments))

            prev_assignments = assignments
            iters -= 1

        return C_final, assignments
                
    def classify(self, data_point):
        return min(range(self.n_clusters), key=lambda p: hmm_distance(data_point, self.centers[p]))