#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cbptools.exceptions import DimensionError
from cbptools.image import img_is_4d, get_masked_series, \
    find_low_variance_voxels, get_f2c_order
from cbptools.connectivity import seed_based_correlation
from cbptools.clean import nuisance_signal_regression, fft_filter
from scipy.signal import detrend
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from nibabel.processing import smooth_image
from fnmatch import fnmatch
from pathlib import Path
from shutil import rmtree
import pandas as pd
import nibabel as nib
import numpy as np
import gc
import os


def connectivity_fmri(time_series: str, seed: str, target: str,
                      participant_id: str, out: str, log_file: str,
                      seed_low_variance: float = 0.05,
                      target_low_variance: float = 0.1,
                      smoothing_fwhm: int = None, confounds: str = None,
                      sep: str = None, usecols: list = None,
                      band_pass: tuple = None, arctanh_transform: bool = True,
                      pca_transform: float = None,
                      compress_output: bool = False) -> None:
    """ Compute a connectivity matrix from functional data.

    Processing steps:
      [1] Apply smoothing to time-series data (optional),
      [2] Apply ROI and target mask to the time-series,
      [3] apply nuisance signal regression on the roi- and target-masked
          time-series separately (optional),
      [4] apply band-pass filtering on the roi- and target-masked
          time-series separately (optional),
      [5] compute the correlation between the roi-masked time-series and
          target-masked time-series,
      [6] apply an arctanh transformation (np.arctanh) on the
          connectivity matrix (optional),
      [7] apply principal component analysis on the connectivity
          matrix (optional)

    Parameters
    ----------
    time_series: str
        Path to the time-series nifti image
    seed: str
        Path to the seed mask nifti image
    target: str
        Path to the target mask nifti image
    participant_id: str
        Unique identifier of the participant currently being processed
    out: str
        Output filename for the connectivity matrix (.npy)
    log_file : str
        Output filename for the log file (.log)
    seed_low_variance : float
        Percentage of low-variance voxels (tolerance of
        np.finfo(np.float32).eps) allowed to be within the seed
    target_low_variance : float
        Percentage of low-variance voxels (tolerance of
        np.finfo(np.float32).eps) allowed to be within the target
    smoothing_fwhm: int, optional
        Smoothing kernel at FWHM in milimeters. If None, smoothing is
        skipped.
    confounds : str, optional
        Path to a tabular confounds file.
    sep : str, optional
        Separator used for reading the confounds file (e.g., .tsv has
        '\t', .csv has ',' or ';')
    usecols : list, optional
        List containing the names of all columns that will be used
        for nuisance signal regression. If a confounds file is given,
        but this argument is not, all columns will be used.
        Wildcards are allowed (e.g., 'motion-*' will include all
        columns that start with 'motion-')
    band_pass: tuple, optional
        Band-pass filter parameters. The following order should be
        maintained for the tuple: (high_pass, low_pass, tr).
        If one is not given, band-pass filtering cannot proceed.
    arctanh_transform: bool, optional
        Apply np.arctanh to the connectivity matrix. This will be
        applied by default.
    pca_transform: float, optional
        Apply sklearn.decomposition.PCA to the connectivity matrix.
        The parameter value is used as the n_components value
        specified in the description of the aforementioned class.
        All other parameters are kept to sklearn defaults. Applying
        this returns the ROI-voxels by principal components as a
        connectivity matrix. If this is set to None, this step will
        be ignored.
    compress_output: bool, optional
        Compress the output connectivity matrices using numpy savez_compressed
        to reduce the size on disk.
    """

    time_series = nib.load(time_series)
    seed_img = nib.load(seed)
    target_img = nib.load(target)

    if not img_is_4d(time_series):
        raise DimensionError(4, len(time_series.shape))

    if smoothing_fwhm is not None:
        time_series = smooth_image(time_series, fwhm=smoothing_fwhm)

    seed_series = get_masked_series(time_series, seed_img)
    target_series = get_masked_series(time_series, target_img)
    del time_series
    gc.collect()

    # Identify low-variance voxels and log them
    in_seed = find_low_variance_voxels(data=seed_series)
    in_target = find_low_variance_voxels(data=target_series)
    bad_seed = in_seed.size / np.count_nonzero(
        seed_img.get_data()) > seed_low_variance
    bad_target = in_target.size / np.count_nonzero(
        target_img.get_data()) > target_low_variance

    pd.DataFrame(
        data=[[participant_id, in_seed, in_target, bad_seed or bad_target]],
        columns=['participant_id', 'low_variance_in_seed',
                 'low_variance_in_target', 'low_variance_excluded']
    ).to_csv(log_file, sep='\t', index=False)

    # If the participant has data exceeding the seed- or target low variance
    # threshold, output an empty file
    if bad_seed or bad_target:
        np.savez(out, connectivity=np.array([]))
        return

    # Nuisance Signal Regression
    if confounds is not None:
        # Fix separator if needed
        if sep is None:
            ext = os.path.splitext(confounds)[-1]
            separators = {'.tsv': '\t', '.csv': ','}
            if ext in separators.keys():
                sep = separators[ext]

        # Check if usecols contains wildcards to extend upon the header
        if usecols is not None:
            usecols = set(usecols)
            header = pd.read_csv(
                confounds,
                sep=sep,
                header=None,
                nrows=1
            ).values.tolist()[0]
            usecols = [x for x in header if any(fnmatch(x, p)
                                                for p in usecols)]

        confounds = pd.read_csv(confounds, sep=sep, usecols=usecols).values
        seed_series = nuisance_signal_regression(
            seed_series,
            confounds=confounds,
            demean=False
        )
        target_series = nuisance_signal_regression(
            target_series,
            confounds=confounds,
            demean=False
        )

    # Apply band-pass filter if high_pass, low_pass, and tr are defined
    high_pass, low_pass, tr = band_pass
    if all([low_pass, high_pass, tr]):
        seed_series = fft_filter(
            seed_series,
            low_pass=low_pass,
            high_pass=high_pass,
            tr=tr
        )
        target_series = fft_filter(
            target_series,
            low_pass=low_pass,
            high_pass=high_pass,
            tr=tr
        )

    connectivity = seed_based_correlation(
        x=seed_series,
        y=target_series,
        standardize=True
    )

    if arctanh_transform:
        # Values at 1 or -1 causing atanh inf's, here we set them slightly
        # below 1 or above -1.
        connectivity[connectivity >= 1] = np.nextafter(
            np.float32(1.),
            np.float32(-1)
        )
        connectivity[connectivity <= -1] = np.nextafter(
            np.float32(-1.),
            np.float32(1)
        )
        connectivity = np.arctanh(connectivity)

    if pca_transform is not None:
        connectivity = detrend(connectivity, axis=1, type='constant')
        pca = PCA(n_components=pca_transform)
        connectivity = pca.fit_transform(connectivity)

    # Ensure float32
    connectivity = connectivity.astype(np.float32)

    if compress_output:
        np.savez_compressed(out, connectivity=connectivity)
    else:
        np.savez(out, connectivity=connectivity)


def merge_connectivity_logs(log_files: list, out: str) -> None:
    df = pd.concat((pd.read_csv(f, sep='\t', index_col=False)
                    for f in log_files), axis=0).reset_index(drop=True)

    df.to_csv(out, sep='\t', index=False)


def validate_connectivity(log_file: str, connectivity: str, labels: str,
                          n_clusters: list, out: str) -> None:
    df = pd.read_csv(log_file, sep='\t')
    bad_participants = df[df['low_variance_excluded']]['participant_id']
    bad_participants = bad_participants.tolist()

    # Remove files with erroneous or no data
    if bad_participants:
        print('--------------------------------------------------------------')
        print(' %s participants with bad data found.' % len(bad_participants))
        print('--------------------------------------------------------------')

        for bad_participant in bad_participants:
            connectivity_file = connectivity.format(
                participant_id=bad_participant
            )
            labels_files = [labels.format(
                participant_id=bad_participant,
                n_clusters=k
            ) for k in n_clusters]

            if os.path.exists(connectivity_file):
                os.remove(connectivity_file)
                print('Removing output file %s' % connectivity_file)

            for file in labels_files:
                if os.path.exists(file):
                    os.remove(file)
                    print('Removing output file %s' % file)

        print('\nOpen log/connectivity_log.tsv to find problematic '
              'participants in the low_variance_excluded column\n')
        raise ValueError('Participants with bad data found. Open '
                         'log/connectivity_log.tsv for more details.')

    # Touch an output file that subsequent rules depend on
    else:
        Path(out).touch()


def connectivity_dmri(fdt_matrix2: str, seed: str, out: str,
                      cleanup_fsl: bool = True, pca_transform: float = None,
                      cubic_transform: bool = False,
                      compress_output: bool = False) -> None:
    """ Compute a connectivity matrix from functional data. This
    method uses FSL's probtrackx2 function which must be accessible
    from the terminal.

    Use '>> probtrackx2 --help' to get more information about the
    input arguments. The optional cubic and pca transformations
    are applied, in the mentioned order, after the connectivity
    matrix has been extracted from the FSL fdt_matrix2 output.

    Parameters
    ----------
    fdt_matrix2 : str
        Path to the probtrackx2 output file for fdt_matrix2
    seed : str
        Path to the seed mask nifti image
    out : str
        Output filename for the connectivity matrix (.npy)
    cleanup_fsl: bool, optional
        Remove the FSL output directory defined in tmp_dir. Once the
        connectivity matrix has been extracted, this data will no
        longer be used by the pipeline.
    pca_transform : float, optional
        Apply sklearn.decomposition.PCA to the connectivity matrix.
        The parameter value is used as the n_components value
        specified in the description of the aforementioned class.
        All other parameters are kept to sklearn defaults. Applying
        this returns the ROI-voxels by principal components as a
        connectivity matrix. If this is set to None, this step will
        be ignored.
    cubic_transform : bool, optional
        Apply a cubic transformation to the connectivity matrix.
    compress_output: bool, optional
        Compress the output connectivity matrices using numpy savez_compressed
        to reduce the size on disk.
    """
    i, j, value = np.loadtxt(fdt_matrix2, unpack=True)
    i = i.astype(int) - 1  # convert to int for indexing
    j = j.astype(int) - 1  # FSL indexes from 1, but we need 0-indexing

    connectivity = coo_matrix((value, (i, j)))
    connectivity = connectivity.todense(order='F')

    if cubic_transform:
        connectivity = np.power(connectivity, 1 / 3)

    if pca_transform is not None:
        connectivity = detrend(connectivity, axis=1, type='constant')
        pca = PCA(n_components=pca_transform)
        connectivity = pca.fit_transform(connectivity)

    # Reorder seed-voxels from F- to C-order
    seed = nib.load(seed)
    reorder = get_f2c_order(seed)
    connectivity = connectivity[reorder, :]

    # Ensure float32
    connectivity = connectivity.astype(np.float32)

    if compress_output:
        np.savez_compressed(out, connectivity=connectivity)
    else:
        np.savez(out, connectivity=connectivity)

    if cleanup_fsl:
        rmtree(os.path.dirname(fdt_matrix2))
