.. |br| raw:: html

    <br/>

.. _TaskClustering:

==========
Clustering
==========
Clustering can be separated into three different tasks, for the three currently available clustering algorithms. Each
task has its own unique set of cluster options. All tasks use the `individual/{participant_id}/connectivity.npz`
connectivity matrix files generated in the previous task, or the user-defined connectivity matrices when the modality
is set to 'connectivity'.

As described for the earlier tasks, the wildcards `{participant_id}` and `{session}` are placeholders. A new wildcard
is added for this task called `{n_clusters}`. This is a placeholder for the number of clusters that are computed for
the cluster labels contained within the file.

KMeans Clustering
=================
This task is active only if the `kmeans` clustering algorithm is defined in the configuration file.

.. glossary::
    Configuration fields
        parameters.clustering.n_clusters |br|
        parameters.clustering.cluster_options.algorithm |br|
        parameters.clustering.cluster_options.init |br|
        parameters.clustering.cluster_options.max_iter |br|
        parameters.clustering.cluster_options.n_init

    Output
        `individual/{participant_id}/{n_clusters}cluster_labels.npy`

    Logging
        `log/{participant_id}.k{n_clusters}.kmeans_clustering.log`

    Benchmarking
        `benchmarks/{participant_id}.k{n_clusters}.kmeans_clustering.log`

This task will apply the k-means clustering algorithm on a connectivity matrix using the `sklearn` package
(`sklearn.cluster.KMeans`) and return the resulting cluster labels.

Spectral Clustering
===================
This task is active only if the `spectral` clustering algorithm is defined in the configuration file.

.. glossary::
    Configuration fields
        parameters.clustering.n_clusters
        parameters.clustering.cluster_options.n_init
        parameters.clustering.cluster_options.kernel
        parameters.clustering.cluster_options.gamma
        parameters.clustering.cluster_options.n_neighbors
        parameters.clustering.cluster_options.assign_labels
        parameters.clustering.cluster_options.degree
        parameters.clustering.cluster_options.coef0
        parameters.clustering.cluster_options.eigen_tol
        parameters.clustering.cluster_options.eigen_solver

    Output
        `individual/{participant_id}/{n_clusters}cluster_labels.npy`

    Logging
        `log/{participant_id}.k{n_clusters}.spectral_clustering.log`

    Benchmarking
        `benchmarks/{participant_id}.k{n_clusters}.spectral_clustering.log`

This task will apply the spectral clustering algorithm on a connectivity matrix using the `sklearn` package
(`sklearn.cluster.SpectralClustering`) and return the resulting cluster labels.

Not all configuration fields listed above will necessarily be used: `gamma` is only used when the
`kernel` is rbf, polynomial, sigmoid, laplacian, or chi2; `n_neighbors` is only used if the `kernel` is
nearest_neighbors; `degree` is only used with a polynomial `kernel`; `coef0` is used only with a polynomial or sigmoid
`kernel`; and `eigen_tol` is only used when the `eigen_solver` is arpack.

.. note::
    Clustering results can vary wildly between the different kernels

.. note::
    Clustering may fail if the `eigen_tol` is set too low

If the clustering fails due to a `numpy.linalg.LinAlgError` or because the requested number of clusters was not
returned, CBPtools will store an empty output file and create a warning in the log file. At a later stage in the
CBPtools workflow, processing will halt and provide a more detailed error log.

Hierarchical Clustering
=======================
This task is active only if the `agglomerative` clustering algorithm is defined in the configuration file.

.. glossary::
    Configuration fields
        parameters.clustering.n_clusters
        parameters.clustering.cluster_options.distance_metric
        parameters.clustering.cluster_options.linkage

    Output
        `individual/{participant_id}/{n_clusters}cluster_labels.npy`

    Logging
        `log/{participant_id}.k{n_clusters}.agglomerative_clustering.log`

    Benchmarking
        `benchmarks/{participant_id}.k{n_clusters}.agglomerative_clustering.log`

This task will apply the agglomerative clustering algorithm on a connectivity matrix using the `sklearn` package
(`sklearn.cluster.AgglomerativeClustering`) and return the resulting cluster labels.

Validating Cluster Labels
=========================
At this point in the workflow the connectivity matrices and cluster labels are computed for all participants. If any of
the participants contains problematic results (i.e., the connectivity or cluster labels file is empty due to an error
during processing), CBPtools will provide a log file at `log/validate_cluster_labels.log` with information about the
participant IDs and reason of the problematic results. Processing will halt at this point, as manual actions are
required (e.g., addressing the issue(s) by removing the participant IDs from `participants.tsv`, or any other action
that can create proper connectivity and cluster label output).

If there are no problems at this point, the workflow will resume with the next tasks.
