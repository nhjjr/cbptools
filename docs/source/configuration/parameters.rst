.. raw:: html

    <style> .green {color:green}  .black {color:black}</style>

.. role:: green
.. role:: black

==========
Parameters
==========

Masking
=======
The mask preprocessing is executed during the setup step (i.e., prior to executing the pipeline). It transforms the
masks based on the parameters used below. Not all parameters are used for each `input_data_type`. The connectivity
input data type has no masking parameters at all, since mask preprocessing is skipped.

All parameters are under the `parameters` key, and the masking parameters are added under the `masking` key:

.. code-block:: yaml

    parameters:
        masking:
            threshold: 0.0
            median_filter: false
            ...


.. glossary::
   :black:`threshold`   *float, default = 0.0, input_data_type =* (:green:`'rsfmri', 'dmri'`)
      Threshold above which voxels in the ROI mask image are defined as 1's. This is only applied if the mask is not
      binary.

   :black:`median_filter`   *bool, default = false, input_data_type =* (:green:`'rsfmri'`)
      Apply median filtering to the seed mask

   :black:`median_filter_dist`  *int, default = 1, input_data_type =* (:green:`'rsfmri'`)
      Median filtering distance

   :black:`del_seed_expand` *int, optional, input_data_type =* (:green:`'rsfmri'`)
      Expand the border around the seed mask (in milimeter) for removal from the target mask. This should only be
      applied if the input time-series data is smoothed, using the smoothing kernel as a value for this parameter.

   :black:`subsample`   *int, optional, input_data_type =* (:green:`'rsfmri'`)
      Apply subsampling to the target mask to improve computational efficiency at minimal loss of specificity. This
      removes every second voxel from the mask and is only recommended if the data has been smoothed.

   :black:`resample_to_mni` *bool, default = true, input_data_type =* (:green:`'rsfmri', 'dmri'`)
      Resample the input masks (seed- and target mask) to match the 2mm MNI152 gray matter mask in shape and affine.

   :black:`upsample_seed_to`    *list, optional, input_data_type =* (:green:`'dmri'`)
      Upsample the seed mask to the specified voxel size (e.g., from [3, 3, 3] as 3mm isotropic to [1, 1, 1] as 1mm
      isotropic). If left empty or as null, no upsampling will be done.

   :black:`upsample_target_to`  *list, optional, input_data_type =* (:green:`'dmri'`)
      Downsample the target mask to the specified voxel size, similar to how upsample_seed_to works.

Connectivity
============
The connectivity task will generate a connectivity matrix from the specified input data.

All parameters are under the `parameters` key, and the connectivity parameters are added under the `connectivity` key:

.. code-block:: yaml

    parameters:
        connectivity:
            pca_transform: 0.95
            high_pass: 0.01
            ...


.. glossary::
   :black:`seed_low_variance_threshold`   *float, default = 0.05, input_data_type =* (:green:`'rsfmri'`)
      When more than this specified percentage of voxels within the seed has low or no variance over the entire time
      course, the processing for this participant will not continue. A detailed error report is provided once all
      connectivity is processed and further processing is halted until the problems are resolved.

   :black:`target_low_variance_threshold`   *float, default = 0.1, input_data_type =* (:green:`'rsfmri'`)
      Same as `seed_low_variance_threshold`, except concerning voxels within the target masked area.

   :black:`high_pass`   *float, optional, input_data_type =* (:green:`'rsfmri'`)
      High-pass value for the band-pass filter. If this is used, `low_pass` and `tr` must also be specified

   :black:`low_pass`    *float, optional, input_data_type =* (:green:`'rsfmri'`)
      Low-pass value for the band-pass filter. If this is used, `high_pass` and `tr` must also be specified

   :black:`tr`  *float, optional, input_data_type =* (:green:`'rsfmri'`)
      Repetition time (in seconds) required for band-pass filtering

   :black:`smoothing_fwhm`  *int, optional, input_data_type =* (:green:`'rsfmri'`)
      FWHM kernel value for smoothing. If left empty, smoothing is skipped

   :black:`arctanh_transform`   *bool, default = true, input_data_type =* (:green:`'rsfmri'`)
      Arctanh transform applied to the connectivity matrix

   :black:`pca_transform`   *float, optional, input_data_type =* (:green:`'rsfmri', 'dmri'`)
      PCA transform applied to the connectivity matrix. This value is equivalent to n_components in
      sklearn.decomposition.PCA

   :black:`dist_thresh` *float, default = 5.0, input_data_type =* (:green:`'dmri'`)
      (probtrackx2) Discards samples shorter than this threshold (in mm)

   :black:`loop_check`  *bool, default = false, input_data_type =* (:green:`'dmri'`)
      (probtrackx2) Perform loopchecks on paths - slower, but allows lower curvature threshold

   :black:`c_thresh`    *float, default = 0.2, input_data_type =* (:green:`'dmri'`)
      (probtrackx2) Curvature threshold

   :black:`step_length` *float, default = 0.5, input_data_type =* (:green:`'dmri'`)
      (probtrackx2) Steplength in mm

   :black:`n_samples`   *int, default = 200, input_data_type =* (:green:`'dmri'`)
      (probtrackx2) Number of samples

   :black:`n_steps` *int, default = 2000, input_data_type =* (:green:`'dmri'`)
      (probtrackx2) Number of steps per sample

   :black:`correct_path_distribution`   *bool, default = false, input_data_type =* (:green:`'dmri'`)
      (probtrackx2) Correct path distribution for the length of the pathways

   :black:`cleanup_fsl` *bool, default = true, input_data_type =* (:green:`'dmri'`)
      Remove all files created by probtrackx2 (except `fdt_matrix2.dot`) after the connectivity matrix has been
      extracted.

   :black:`cubic_transform` *bool, default = true, input_data_type =* (:green:`'dmri'`)
      Apply a cubic transformation on the connectivity matrix

Clustering
==========
Connectivity matrices will be clustered using k-means (sklearn.cluster.KMeans) and these clustering results will
subsequently be clustered using hierarchical clustering to obtain a group parcellation.

All parameters are under the `parameters` key, and the clustering parameters are added under the `clustering` key:

.. code-block:: yaml

    parameters:
        clustering:
            algorithm: auto
            group_method: mode
            init: random
            ...


.. glossary::
   :black:`n_clusters`   *list, input_data_type =* (:green:`'rsfmri', 'dmri', 'connectivity'`)
      A list of cluster numbers to be evaluated (entered as [2, 3, 8] to receive a 2, 3, and 8-cluster solution)

   :black:`algorithm`   *str, default = 'auto', input_data_type =* (:green:`'rsfmri', 'dmri', 'connectivity'`)
      (sklearn.cluster.KMeans) K-means algorithm to use

   :black:`init`   *str, default = 'random', input_data_type =* (:green:`'rsfmri', 'dmri', 'connectivity'`)
      (sklearn.cluster.KMeans) Method for initialization

   :black:`max_iter`   *int, default = 10000, input_data_type =* (:green:`'rsfmri', 'dmri', 'connectivity'`)
      (sklearn.cluster.KMeans) Maximum number of iterations of the k-means algorithm for a single run.

   :black:`n_init`   *int, default = 256, input_data_type =* (:green:`'rsfmri', 'dmri', 'connectivity'`)
      (sklearn.cluster.KMeans) Number of time the k-means algorithm will be run with different centroid seeds.

   :black:`linkage`   *str, default = 'complete', input_data_type =* (:green:`'rsfmri', 'dmri', 'connectivity'`)
      (scipy.cluster.hierarchy.linkage) The linkage algorithm to use (allowed values: 'single', 'average', 'complete')

   :black:`group_method`   *str, default = 'agglomerative', input_data_type =* (:green:`'rsfmri', 'dmri', 'connectivity'`)
      Method for obtaining group-level clustering results (allowed values: 'agglomerative', 'mode')

   :black:`internal_validity_metrics`   *list, default = ['silhouette_score'], input_data_type =* (:green:`'rsfmri', 'dmri', 'connectivity'`)
      List of internal validity metrics to assess (allowed values: ['silhouette_score',
      'davies_bouldin_score', 'calinski-harabasz'])

   :black:`similarity_metric`   *str, default = 'adjusted_rand_score', input_data_type =* (:green:`'rsfmri', 'dmri', 'connectivity'`)
      Similarity metric to use to generate between-subject cluster comparisons and subject to group-level cluster
      comparisons (allowed values: 'adjusted_rand_score', 'adjusted_mutual_info_score', 'v_measure_score')


Summary
=======
Parameters for generating the summary results.

All parameters are under the `parameters` key, and the summary parameters are added under the `summary` key:

.. code-block:: yaml

    parameters:
        summary:
            algorithm: auto
            ...


.. glossary::
   :black:`figure_format`   *str, default = 'png', input_data_type =* (:green:`'rsfmri', 'dmri', 'connectivity'`)
      Format of the output figures generated for the summary (allowed values: 'png', 'svg', 'pdf', 'ps', 'eps').


References
==========
References to external documentation:

   * `probtrackx2 <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide#PROBTRACKX_-_probabilistic_tracking_with_crossing_fibres)>`_
   * `sklearn.cluster.KMeans <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)>`_
   * `scipy.cluster.hierarchy.linkage <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)>`_
