#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
import numpy as np


def main(connectivity, file: str, n_clusters: int, algorithm: str='auto', init: str='random', max_iter: int=10000,
         n_init: int=100):
    """ Perform k-means clustering on the input connectivity matrix.

    Parameters
    ----------
    connectivity : str
        Path to the connectivity matrix that should be clustered (.npy)
    file : str
        Output filename for the k-means labels (.npy)
    n_clusters : int
        The number of clusters to form. See sklearn.cluster.KMeans
    algorithm : str, optional
        K-means algorithm to use. See sklearn.cluster.KMeans
    init : str, optional
        Method for initialization, defaults to ‘random’. See sklearn.cluster.KMeans
    max_iter : int, optional
        Number of iterations of the k-means algorithm. See sklearn.cluster.KMeans
    n_init : int
        Number of initializations of the k-means algorithm. See sklearn.cluster.KMeans
    """

    connectivity = np.load(connectivity)
    kmeans = KMeans(algorithm=algorithm, init=init, max_iter=max_iter, n_clusters=n_clusters, n_init=n_init)
    kmeans.fit(connectivity)
    np.save(file, kmeans.labels_)  # cluster labels are 0-indexed


if __name__ == '__main__':
    main(
        connectivity=snakemake.input[0],
        file=snakemake.output[0],
        n_clusters=snakemake.params.get('n_clusters'),
        algorithm=snakemake.params.get('algorithm'),
        init=snakemake.params.get('init'),
        max_iter=snakemake.params.get('max_iter'),
        n_init=snakemake.params.get('n_init')
    )
