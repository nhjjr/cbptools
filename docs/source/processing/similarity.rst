.. |br| raw:: html

    <br/>

.. _TaskSimilarity:

==========
Similarity
==========
As described for the earlier tasks, the wildcards `{participant_id}` is a placeholder.

.. glossary::
    Configuration fields
        data.references (optional) |br|
        parameters.clustering.validity.similarity
        parameters.clustering.n_clusters

    Output
        `group/{n_clusters}clusters/individual_similarity.npy` |br|
        `group/group_similarity.tsv` |br|
        `group/cophenetic_correlation.tsv` |br|
        `group/reference_similarity.tsv` (optional)

    Benchmarking
        `benchmarks/k{n_clusters}.individual_similarity.log` |br|
        `benchmarks/group_similarity.log` |br|
        `benchmarks/reference_similarity.log` (optional)


Individual-to-Individual Similarity
===================================
Using the individual participant cluster labels per cluster granularity *k*, a similarity matrix is computed containing
the similarity scores between the cluster labels of each participant using the defined similarity metric. The `sklearn`
package is used for all available metrics (`sklearn.metrics`) -- adjusted Rand index, adjusted mutual information
score, and the V measure score.

Individual-to-Group Similarity
==============================
Using the same individual participant cluster labels and the group-level cluster labels, the similarity is computed
between the individual participant cluster labels and the group-level cluster labels per cluster granularity *k*. This
is done using the same approach as the individual-to-individual similarity.

At this point the cophenetic correlation, computed in the :ref:`previous <TaskGrouping>` task is merged.

Reference-to-Group Similarity
=============================
If reference images are provided, the labeled voxels are extracted from the images. The defined similarity metric is
computed between each reference and each group-clustering result, done using the same apparoch as the
individual-to-individual similarity.
