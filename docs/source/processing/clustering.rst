.. _TaskClustering:

==========
Clustering
==========
The connectivity matrices are passed to the clustering task. If connectivity matrices are given directly as input, then
this is the initial task. The clustering task is likewise parallelized so that each subject and each clustering
granularity *k* (i.e., the list of requested cluster numbers) can be run simultaneously.

The connectivity matrices are loaded and the k-means algorithm is applied (using sklearn.cluster.KMeans). This results
in a set of labels (solutions) where each seed voxel is assigned to a cluster, for each clustering granularity *k* and
each subject.