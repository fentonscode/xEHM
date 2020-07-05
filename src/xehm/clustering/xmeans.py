# X-Means cluster classifier
#

from .classifier import Classifier
from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from scipy.spatial import distance

__all__ = ["XMeans"]


class XMeans(Classifier):
    def __init__(self):
        super().__init__()
        self.labels = []

    def assign(self, data: np.ndarray, parameters: Tuple = None):

        # Cluster to branch the emulators
        iter_limit = data.shape[0]
        b_trace = []
        k = 1
        while k < iter_limit:
            clusters = KMeans(n_clusters=k).fit(data)
            chunks = [data[np.where(clusters.labels_ == i)] for i in range(k)]
            bic = _bic(chunks, clusters.cluster_centers_)
            b_trace.append(bic)
            if k == 1:
                prev_bic = bic
                k += 1
                continue
            if bic > prev_bic:
                break
            prev_bic = bic
            k += 1

        self.n_groups = k
        self.centres = clusters.cluster_centers_
        self.labels = clusters.labels_
        return self

    @staticmethod
    def compute_bic(clusters, x):

        # assign centers and labels
        centers = [clusters.cluster_centers_]
        labels = clusters.labels_
        # number of clusters
        m = clusters.n_clusters
        # size of the clusters
        n = np.bincount(labels)
        # size of data set
        N, d = x.shape

        # compute variance for all clusters beforehand
        cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(x[np.where(labels == i)], [centers[0][i]],
                                                               'euclidean') ** 2) for i in range(m)])

        const_term = 0.5 * m * np.log(N) * (d + 1)

        BIC = np.sum([n[i] * np.log(n[i]) -
                      n[i] * np.log(N) -
                      ((n[i] * d) / 2) * np.log(2 * np.pi * cl_var) -
                      ((n[i] - 1) * d / 2) for i in range(m)]) - const_term

        return (BIC)


# Bayesian information criterion
# NOTE: There are so many incorrect formulas on the internet about this
#
# BIC = kln(n) - 2ln(L): Accept no substitutes!
#
def _bic(clusters, centroids):
    num_points = sum(len(cluster) for cluster in clusters)

    # Numpy completely butchers 1D arrays/slices
    num_dims = clusters[0].shape[1] if clusters[0].ndim > 1 else 1

    log_likelihood = _loglikelihood(num_points, num_dims, clusters, centroids)
    num_params = _free_params(len(clusters), num_dims)
    return -(log_likelihood - num_params / 2.0 * np.log(num_points))


# Why is this +1?
def _free_params(num_clusters, num_dims):
    return num_clusters * (num_dims + 1)


def _loglikelihood(num_points, num_dims, clusters, centroids):
    ll = 0
    for cluster in clusters:
        fRn = len(cluster)
        t1 = fRn * np.log(fRn)
        t2 = fRn * np.log(num_points)
        variance = max(
            _cluster_variance(num_points, clusters, centroids),
            np.nextafter(0, 1))
        t3 = ((fRn * num_dims) / 2.0) * np.log((2.0 * np.pi) * variance)
        t4 = num_dims * (fRn - 1.0) / 2.0
        ll += t1 - t2 - t3 - t4
    return ll


#
# clusters = list of points in each cluster
# centriods = array of centre locations
#
def _cluster_variance(num_points, clusters, centroids) -> float:
    s: float = 0
    num_dims = clusters[0].shape[1] if clusters[0].ndim > 1 else 1
    denom: float = float(num_points - len(centroids)) * num_dims
    for cluster, centroid in zip(clusters, centroids):
        distances = euclidean_distances(cluster, centroid.reshape(-1, 1))
        s += (distances * distances).sum()
    return s / denom