.. |br| raw:: html

    <br/>

.. raw:: html

    <style>
        .green { color: green; font-weight: bold }
    </style>

.. role:: green

.. _TaskConnectivitydMRI:

=================
dMRI Connectivity
=================
The wildcards `{participant_id}` and `{session}` are placeholders for the ID of the participant in the data set, and
the session string defined in the :green:`data:session` field, respectively. This applies to the output, as well as the
logging and benchmark output files.

This task can be I/O heavy if bedpostX output images are large and have to be accessed from a remote source. To
mitigate the effects on computation time, it is possible to define the `io` parameter in the `--resources` directive of
snakemake. This task is given 1 `io` token, so if `snakemake --resources io=10` is used, then at most 10 connectivity
tasks can be run in parallel.

.. glossary::
    Configuration fields
        data.masks.seed |br|
        data.masks.target (optional) |br|
        data.bet_binary_mask |br|
        data.xfm |br|
        data.inv_xfm |br|
        data.samples |br|
        parameters.connectivity.dist_thresh |br|
        parameters.connectivity.c_thresh |br|
        parameters.connectivity.step_length |br|
        parameters.connectivity.n_samples |br|
        parameters.connectivity.n_steps |br|
        parameters.connectivity.loop_check |br|
        parameters.connectivity.correct_path_distribution |br|
        parameters.connectivity.cubic_transform (optional) |br|
        parameters.connectivity.pca_transform (optional) |br|
        parameters.report.compress_output (optional) |br|
        parameters.connectivity.cleanup_fsl (optional)

    Output
        `individual/{participant_id}/probtrackx2` (temporary folder) |br|
        `individual/{participant_id}/probtrackx2_{session}` (temporary folder; multi-session) |br|
        `individual/{participant_id}/connectivity.npz` |br|
        `individual/{participant_id}/connectivity_{session}.npz` (temporary file; multi-session)

    Logging
        `log/{participant_id}.connectivity_dmri.log` (single-session) |br|
        `log/{participant_id}.{session}.connectivity_dmri.log` (multi-session) |br|
        `log/{participant_id}.merge_sessions.log` (multi-session)

    Benchmarking
        `benchmarks/{participant_id}.probtrackx2.log` (single-session) |br|
        `benchmarks/{participant_id}.{session}.probtrackx2.log` (multi-session) |br|
        `benchmarks/{participant_id}.connectivity_dmri.log` (single-session) |br|
        `benchmarks/{participant_id}.{session}.connectivity_dmri.log` (multi-session) |br|
        `benchmarks/{participant_id}.merge_sessions.log` (multi-session)

Probabilistic Diffusion Tractography
====================================
Using FSL's `probtrackx2` a tractography analysis is run which produces sample streamlines. For more information, see
the `probtrackx2 documentation <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide#PROBTRACKX_-_probabilistic_tracking_with_crossing_fibres>`_.
This produces a folder with various output files, of which the `fdt_matrix2.dot` file is used for further processing.

The `fdt_matrix2.dot` file is a sparse FDT matrix in F-contiguous order, which is densified using the `scipy` package
(`scipy.sparse.coo_matrix` and the `.todense()` method). This matrix is now referred to as a connectivity matrix.

Transforms (optional)
=====================
A cubic transform is optionally applied to the connectivity matrix using the `numpy` package (`numpy.power`) and a
power of 1/3.

Furthermore optional, a principal component analysis (PCA) transform can be applied to the connectivity matrix using
the `sklearn` package (`sklearn.decomposition.PCA`). First, the `scipy` package is used for detrending
(`scipy.signal.detrend`, using the 'constant' type of detrending). PCA then reduces the number of target features in
the connectivity matrix based on the value entered as the `components` in the configuration file. If a component value
below 1 is used, then a number of components will be returned explaining that much of the variance (i.e., for 0.95, the
components explaining 95% of the variance are returned as target features). If a number at or above 1 is used, that
many components are returned.

Reordering the Matrix
=====================
Since the `fdt_matrix2.dot` is in F-contiguous order, the connectivity matrix will be reordered to C-contiguous order.
This keeps all the CBPtools output files consistent in terms of their ordering. This is done using the
`cbptools.image.get_f2c_order` method, which provides reordering indices using the seed mask such that an F extraction
order is turned into a C extraction order. This new order is applied to the x-axis (seed voxels) of the connectivity
matrix, but not the y-axis (as the target value ordering is not important for the remaining procedures in the
*CBPtools* workflow).

Merge Sessions
==============
If multi-session input data is used, then each participant will provide multiple connectivity matrices. These matrices
are averaged and the result is used as the one connectivity matrix for that participant. Note that if multi-session
data is being used, then the (optional) PCA transformation will instead be performed after the sessions have been
averaged. This is necessary, as each session may return a different number of components causing the averaging to fail.
