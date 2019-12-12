.. _TaskGrouping:

========
Grouping
========
All subject clustering solutions are combined into one matrix, sorted by the participant_id order in the participants
file (which is alphanumerically sorted). The pairwise hamming distance (y) is calculated on the matrix (x). Then,
hierarchical clustering is performed on this matrix (z) with the linkage algorithm specified in the configuration file.
The cophenetic correlation is then calculated between z and y. The tree is then cut at the requested cluster number to
obtain a reference clustering.

.. code-block:: python

    y = pdist(x, metric='hamming')
    z = hierarchy.linkage(y, method=linkage, metric='hamming')
    coph = hierarchy.cophenet(z, y)
    group_labels = hierarchy.cut_tree(z, n_clusters=len(np.unique(x)))

This reference clustering can be chosen as the group level cluster solution if so
specified in the configuration file (`method = agglomerative`). However, by default it is only used as a reference for
relabeling the subject clustering solutions (`method = mode`). This relabeling is necessary as the number used for
identifying a cluster may differ between subjects. The relabeling that most matches the reference solution is then kept.

The relabeled solutions are again combined into one matrix, and the mode (np.mode) is taken and used as the group level
solution. Lastly, the group level solution is mapped onto the seed mask for each clustering granularity *k* and stored
as a NIfTI image. This allows the clustering results to be viewed using any NIfTI image viewer.