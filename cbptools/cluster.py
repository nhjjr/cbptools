#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for working with clustering results"""

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
import itertools
import numpy as np


def relabel(reference: np.ndarray, x: np.ndarray) -> (np.ndarray, list):
    """Relabel cluster labels to best match a reference"""

    if all(np.unique(x) != np.unique(reference)):
        raise ValueError('Reference and target labels have different cluster indices')

    permutations = itertools.permutations(np.unique(reference))
    accuracy = 0.
    relabeled = None

    for permutation in permutations:
        d = dict(zip(np.unique(x), permutation))
        y = np.zeros(x.shape).astype(int)

        for k, v in d.items():
            y[x == k] = v

        _accuracy = np.sum(y == reference) / len(reference)

        if _accuracy > accuracy:
            accuracy = _accuracy
            relabeled = y.copy()

    return relabeled, accuracy


def find_centers(x: np.ndarray, labels: list) -> np.ndarray:
    """Find the cluster centers of x with a specific set of labels.

    Parameters
    ----------
    x : np.ndarray
        connectivity matrix or pairwise distance matrix
    labels : list
        labels computed by a clustering algorithm (e.g., kmeans)

    Returns
    -------
    np.ndarray
        Cluster Centers of X given labels

    """
    return np.asarray([np.mean(x[labels == i], axis=0) for i in np.unique(labels)])


def weak_deletion_stability_score(x: np.ndarray, y: list, squared=True) -> float:
    """ Internal validity metric

    Parameters
    ----------
    x : np.ndarray
        connectivity matrix or pairwise distance matrix (samples by features)
    y : list
        labels computed by a clustering algorithm (e.g., kmeans)
    squared : bool, optional
        If true, uses sum of squares of k-nearest neighbor distances. Otherwise uses sum.

    Returns
    -------
    float
        weak deletion stability score

    """
    centers = find_centers(x, y)
    reference = within_ss(x, centers, squared)
    res = []

    for k in range(len(centers)):
        tmp_centers = np.delete(centers, k, axis=0)
        res = np.append(res, within_ss(x, tmp_centers, squared))

    return np.max(res/reference)


def within_ss(x: np.ndarray, centers: np.ndarray, squared=True) -> float:
    """ Within sum of squares of centers given x

    Parameters
    ----------
    x : np.ndarray
        connectivity matrix or pairwise distance matrix
    centers : np.ndarray
        cluster centers (k by features)
    squared : bool, optional
        If true, return the sum of squares. Otherwise, return the sum.

    Returns
    -------
    float
        Sum (of squares) of k-nearest neighbor Euclidean distances

    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(centers)
    distances, indices = nbrs.kneighbors(x)
    ss = np.sum(distances ** 2) if squared else np.sum(distances)
    return float(ss)


def davies_bouldin_score(x: np.ndarray, y: list) -> float:
    """ Calculate Davies-Bouldin Index for connectivity matrix x and cluster labels y.

    Based on Matlab's implementation. The Davies-Bouldin index is based on a ratio of within- and between-cluster
    distances, and can be defined as:
        DB = (1/|k|) sum{for i in 1:|k|} max[i!=j] {Di,j}

    Di,j is the within-to-between cluster distance ratio for the ith and jth clusters, defined as:
        Di,j = (di+dj)di,j

    di is the average distance between each point in the ith cluster and the centroid of the ith cluster.
    dj is the average distance between each point in the jth cluster and the centroid of the jth cluster.
    di,j is the Euclidean distance between the centroids of the ith and jth clusters.

    The maximum value of Di,j represents the worst-case within-to-between cluster ratio for cluster i. The optimal
    clustering solution has the smallest Davies-Bouldin index value.

    Parameters
    ----------
    x : np.ndarray
        connectivity matrix or pairwise distance matrix
    y : list
        labels computed by a clustering algorithm (e.g., kmeans)

    Returns
    -------
    float
        Davies-Bouldin index value

    """
    n = np.unique(y)
    k = len(n)
    c = [np.mean(x[y == i], axis=0) for i in n]
    d = [np.mean(np.linalg.norm(x[y == j] - c[i], axis=1, ord=2)) for i, j in enumerate(n)]
    db = np.zeros((k, k))
    db[np.triu_indices(k, 1)] = [(d[i] + d[j]) / euclidean(c[i], c[j]) for i, j in itertools.combinations(range(k), 2)]
    db = float(np.mean(np.max(db+db.T, axis=0)))
    return db


def gap_score(x: np.ndarray, kmeans: KMeans, n_refs: int) -> float:
    """ Calculate the gap value of x, labels, and centroids given a number of references (n_refs)
    The references are generated using kmeans_options keyword arguments, where each keyword should be equivalent to
    the sklearn.cluster.KMeans arguments.

    Return the gap score as a float

    Reference: Tibshirani, Walther, Hastie, (2001). Estimating the number of clusters in a data set via the gap
      statistic. J. R. Statist. Soc. B, 63(2), 411-423
    """

    def _dispersion(x: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Calculate the dispersion between actual points and their assigned centroids"""
        return np.sum(np.sum([np.abs(inst - centroids[label]) ** 2 for inst, label in zip(x, labels)]))

    # Holder for reference dispersion results
    dispersion = _dispersion(x=x, labels=kmeans.labels_, centroids=kmeans.cluster_centers_)
    ref_dispersions = np.zeros(n_refs)

    # For n_references, generate random sample and perform kmeans getting resulting dispersion of each loop
    for i in range(n_refs):
        random_data = np.random.random_sample(size=x.shape)
        kmeans.fit(random_data)
        dispersion = _dispersion(x=random_data, labels=kmeans.labels_, centroids=kmeans.cluster_centers_)
        ref_dispersions[i] = dispersion

    # Calculate gap score
    return np.log(np.mean(ref_dispersions)) - np.log(dispersion)
