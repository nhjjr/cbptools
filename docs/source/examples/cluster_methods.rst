.. _ExampleClusterMethods:

===============
Cluster Methods
===============
CBPtools currently supports three clustering algorithms: k-means, spectral, and agglomerative clustering. The k-means
clustering algorithm is the default, hence when using the `cbptools example` directive, an example file will be
generated containing the k-means clustering default options. Below are examples of how the different clustering
algorithms can be defined, each with their own set of unique options.

KMeans clustering
==================

.. code-block:: yaml

    ...

    parameters:
        clustering:
            method: kmeans
            n_clusters: [2, 3, 4, 5]
            cluster_options:
                algorithm: auto
                init: k-means++
                max_iter: 10000
                n_init: 256

    ...

Spectral clustering
===================

.. code-block:: yaml

    ...

    parameters:
        clustering:
            method: spectral
            n_clusters: [2, 3, 4, 5]
            cluster_options:
                n_init: 256
                kernel: nearest_neighbors
                assign_labels: kmeans
                eigen_solver: arpack
                eigen_tol: 1.0e-5

    ...

Agglomerative clustering
========================

.. code-block:: yaml

    ...

    parameters:
        clustering:
            method: agglomerative
            n_clusters: [2, 3, 4, 5]
            cluster_options:
                distance_metric: euclidean
                linkage: ward

    ...