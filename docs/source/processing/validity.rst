.. _TaskValidity:

========
Validity
========
The requested validity metrics are applied per participant, likewise in paralllel fashion. All clustering solutions of
a given subject (i.e., for each clustering granularity *k*) are processed together. That is, the parallelization is only
across subjects.

The requested validity metrics are each computed per subject using the connectivity matrix as a feature array, and the
predicted labels (solutions) for the participant. Note that for the Silhouette score, the metric for calculating
distance between instances in the feature array is Euclidean.

Once this task is completed for each participant, the resulting scores are merged into a table and used to generate
boxplots for each validity metric and each requested clustering granularity *k*.