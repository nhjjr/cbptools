.. |br| raw:: html

    <br/>

.. _TaskConnectivityrsfMRI:

===================
rsfMRI Connectivity
===================
The wildcards `{participant_id}` and `{session}` are placeholders for the ID of the participant in the data set, and
the session string defined in hte `data:session` field, respectively. This applies to the output, as well as the
logging and benchmark output files.

This task can be I/O heavy if time-series images are large and have to be accessed from a remote source. To mitigate
the effects on computation time, it is possible to define the `io` parameter in the `--resources` directive of
snakemake. This task is given 1 `io` token, so if `snakemake --resources io=10` is used, then at most 10 connectivity
tasks can be run in parallel.

.. glossary::
    Configuration fields
        data.masks.seed |br|
        data.masks.target |br|
        data.time_series |br|
        data.confounds (optional) |br|
        data.session (optional) |br|
        parameters.report.compress_output (optional) |br|
        parameters.connectivity.smoothing (optional) |br|
        parameters.connectivity.low_variance_error |br|
        parameters.connectivity.band_pass_filtering (optional) |br|
        parameters.connectivity.pca_transform (optional) |br|
        parameters.connectivity.arctanh_transform (optional)

    Output
        `individual/{participant_id}/connectivity.npz` |br|
        `individual/{participant_id}/connectivity_{session}.npz` (temporary file; multi-session)

    Logging
        `log/{participant_id}.connectivity_rsfmri.log` (single-session) |br|
        `log/{participant_id}.{session}.connectivity_rsfmri.log` (multi-session) |br|
        `log/{participant_id}.merge_sessions.log` (multi-session)

    Benchmarking
        `benchmarks/{participant_id}.connectivity_rsfmri.log` (single-session) |br|
        `benchmarks/{participant_id}.{session}.connectivity_rsfmri.log` (multi-session) |br|
        `benchmarks/{participant_id}.merge_sessions.log` (multi-session)

Smoothing (optional)
====================
The time-series are optionally smoothed using the `nibabel` package (`nibabel.processing.smooth_image`) and a
FWHM value in *mm* over which to smooth. CBPtools uses the default smoothing mode, 'nearest', as it is the
recommended choice for smoothing. For more information on the smoothing function, read the
`nibabel documentation <https://nipy.org/nibabel/reference/nibabel.processing.html#nibabel.processing.smooth_image>`_.

Masking
=======
The seed and target masks are applied separately to the time-series, resulting in a seed-masked time-series and
target-masked time-series matrix (i.e., seed voxels by timepoints, and target voxels by timepoints, respectively).

Next, the variance for each masked voxel is calculated to identify low-variance voxels. Voxels with a variance below
a tolerance value of `numpy.finfo(np.float32).eps` are marked as low-variance voxels. Since the calculation to obtain
the connectivity matrix includes a division by the standard deviation, low-variance voxels will return `inf` or `NaN`
values. These values will be set to 0. The low-variance error thresholds defined in the configuration file are used to
check whether there are too many low-variance voxels in the seed- or target-masked time-series. If this is the case,
the connectivity computation is aborted and an empty connectivity matrix is returned. The log file will make mention of
this. At a later stage in the CBPtools workflow, processing will halt and provide a more detailed error log.

Nuisance Signal Regression (optional)
=====================================
If a confounds file is provided, the defined columns (all columns if none are defined) will be linearly regressed out
of the time-series signal for both the seed-masked and target-masked time-series. The code snippet below shows how this
is applied using the `numpy` package.

.. code-block:: python

    import numpy as np
    time_series = time_series - np.dot(confounds, np.linalg.lstsq(confounds, data, rcond=-1)[0])

Band-pass Filtering (optional)
==============================
A fast-fourier transform is optionally applied on the seed- and target-masked time-series separately, using the defined
filtering band and repetition time.

Compute Connectivity
====================
The seed-based correlation is computed using the seed- and target-masked time-series and a ddof of 0, resulting in a
connectivity matrix. The code snippet below shows how the connectivity matrix is computed using the `numpy` package.

.. code-block:: python

    import numpy as np

    # Standardization
    x, y = map(lambda z: (z - np.mean(z, axis=0)) / np.std(z, axis=0, ddof=ddof), (x, y))

    # Correlation
    r = (y.T.dot(x) / x.shape[0]).T.astype(np.float32)

Next, all values that are `NaN` or `inf` are set to 0, all values at or above 1 are set slightly lower than 1, and all
values at or below -1 are set slightly higher than -1. This accommodates the (optional) ArcTanh transform, which would
otherwise return `inf` values for values at 1 or -1.

Transforms (optional)
=====================
Using the `numpy` package, optionally an ArcTanh transform is applied on the connectivity matrix.

Furthermore optional, a principal component analysis (PCA) transform can be applied to the connectivity matrix using
the `sklearn` package (`sklearn.decomposition.PCA`). First, the `scipy` package is used for detrending
(`scipy.signal.detrend`, using the 'constant' type of detrending). PCA then reduces the number of target features in
the connectivity matrix based on the value entered as the `components` in the configuration file. If a component value
below 1 is used, then a number of components will be returned explaining that much of the variance (i.e., for 0.95, the
components explaining 95% of the variance are returned as target features). If a number at or above 1 is used, that
many components are returned.

Merge Sessions
==============
If multi-session input data is used, then each participant will provide multiple connectivity matrices. These matrices
are averaged and the result is used as the one connectivity matrix for that participant. Note that if multi-session
data is being used, then the (optional) PCA transformation will instead be performed after the sessions have been
averaged. This is necessary, as each session may return a different number of components causing the averaging to fail.
