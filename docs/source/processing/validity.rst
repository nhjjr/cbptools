.. |br| raw:: html

    <br/>

.. _TaskValidity:

========
Validity
========
As described for the earlier tasks, the wildcards `{participant_id}` is a placeholder.

.. glossary::
    Configuration fields
        parameters.clustering.validity.internal

    Output
        `individual/{participant_id}/internal_validity.tsv` (temporary file) |br|
        `individual/internal_validity.tsv`

    Benchmarking
        `benchmarks/{participant_id}.internal_validity.log` |br|
        `benchmarks/merge_internal_validity.log`

This task uses the connectivity matrix and cluster labels for each participant to compute the requested validity
metrics. The `sklearn` package (`sklearn.metrics`) is used to obtain the Silhouette and Calinski-Harabasz scores,
whereas the Davies-Bouldin score is implemented in `cbptools.cluster.davies_bouldin_score`.

The requested validity metrics are each computed per subject using the connectivity matrix as a feature array, and the
predicted labels (solutions) for the participant. Note that for the Silhouette score, the metric for calculating
distance between instances in the feature array is Euclidean.

Once this task is completed for each participant, the resulting scores are merged into a tab-delimited file.
