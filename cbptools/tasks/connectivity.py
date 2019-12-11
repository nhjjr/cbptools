#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ..image import (img_is_4d, get_masked_series, find_low_variance_voxels,
                     get_f2c_order)
from ..connectivity import seed_based_correlation
from ..clean import nuisance_signal_regression, fft_filter
from ..utils import get_logger
from scipy.signal import detrend
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from nibabel.processing import smooth_image
from fnmatch import fnmatch
from shutil import rmtree
import pandas as pd
import nibabel as nib
import numpy as np
import gc
import os


def connectivity_rsfmri(input: dict, output: dict, params: dict,
                        log: list) -> None:
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
    input : dict
        Input files, allowed: {time_series, seed, target, confounds}
    output : dict
        Output files, allowed {connectivity}
    params : dict
        Parameters, allowed {arctanh_transform, pca_transform, bandpass,
        low_variance_error, compress, confounds, smoothing_fwhm}. For more
        information, see the CBPtools documentation on readthedocs.io under
        the parameters section for 'time_series_proc'.
    log : list
        Log files
    """

    # input, output, params
    time_series_file = input.get('time_series')
    seed_img_file = input.get('seed_mask')
    target_img_file = input.get('target_mask')
    confounds_file = input.get('confounds', None)
    connectivity_file = output.get('connectivity')
    log_file = log[0]
    smoothing = params.get('smoothing', False)
    lv_correction = params.get('low_variance_correction', False)
    lv_in_seed = params.get('low_variance_in_seed', None)
    lv_in_target = params.get('low_variance_in_target', None)
    confounds_sep = params.get('confounds_delimiter', None)
    confounds_cols = params.get('confounds_columns', None)
    bandpass = params.get('bandpass', False)
    pca_transform = params.get('pca_transform', False)
    arctanh_transform = params.get('arctanh_transform', False)
    compress = params.get('compress', False)

    # Set up logging
    logger = get_logger('connectivity_rsfmri', log_file)

    # Load input data
    time_series = nib.load(time_series_file)
    seed_img = nib.load(seed_img_file)
    target_img = nib.load(target_img_file)

    if not img_is_4d(time_series):
        logger.error('%s has incompatible dimensionality: Expected dimension'
                     'is %sD but a %sD image was provided'
                     % (time_series_file, 4, len(time_series.shape)))
        np.savez(connectivity_file, connectivity=np.array([]))
        return

    if smoothing:
        logger.info('applying smoothing (fwhm=%s) to %s'
                    % (smoothing, time_series_file))
        time_series = smooth_image(time_series, fwhm=smoothing)

    seed_series = get_masked_series(time_series, seed_img)
    target_series = get_masked_series(time_series, target_img)
    del time_series
    gc.collect()

    # Identify low-variance voxels and log them
    in_seed = find_low_variance_voxels(data=seed_series)
    in_target = find_low_variance_voxels(data=target_series)
    bad_seed = in_seed.size / np.count_nonzero(seed_img.get_data())
    bad_seed = bad_seed > lv_in_seed
    bad_target = in_target.size / np.count_nonzero(target_img.get_data())
    bad_target = bad_target > lv_in_target

    if in_seed > 0:
        logger.warning('%s low variance seed voxels found in %s'
                       % (in_seed, time_series_file))

    if in_target > 0:
        logger.warning('%s low variance target voxels found in %s'
                       % (in_seed, time_series_file))

    if lv_correction:
        if bad_seed:
            logger.error('number of low variance voxels in seed exceeds the'
                         'threshold (%s) for %s'
                         % (lv_in_seed, time_series_file))

        if bad_target:
            logger.error('number of low variance voxels in target exceeds the'
                         'threshold (%s) for %s'
                         % (lv_in_target, time_series_file))

        if bad_seed or bad_target:
            np.savez(connectivity_file, connectivity=np.array([]))
            return

    if in_target > 0 or in_seed > 0:
        # setting to 0 is done in the seed_based_correlation() method
        logger.warning('low variance voxels will be set to zero')

    # Nuisance Signal Regression
    if confounds_file:
        if confounds_sep is None:
            # Fix delimiter
            ext = os.path.splitext(confounds_file)[-1]
            separators = {'.tsv': '\t', '.csv': ','}
            if ext in separators.keys():
                confounds_sep = separators[ext]

        # Check if usecols contains wildcards to extend upon the header
        if confounds_cols is not None:
            confounds_cols = set(confounds_cols)
            header = pd.read_csv(
                confounds_file,
                sep=confounds_sep,
                header=None,
                nrows=1
            )
            header = header.values.tolist()[0]
            confounds_cols = [
                x for x in header
                if any(fnmatch(x, p) for p in confounds_cols)
            ]

        confounds = pd.read_csv(
            confounds_file, sep=confounds_sep, usecols=confounds_cols
        )

        if confounds_cols is None:
            logger.info('%s nuisance signal regression using all columns'
                        % time_series_file)
        else:
            logger.info('%s nuisance signal regression using: %s'
                        % (time_series_file,
                           str(confounds_cols).strip('[]').replace('\'', '')))

        confounds = confounds.values
        seed_series = nuisance_signal_regression(
            seed_series, confounds=confounds, demean=False)
        target_series = nuisance_signal_regression(
            target_series, confounds=confounds, demean=False)

    # Apply band-pass filter if high_pass, low_pass, and tr are defined
    if bandpass:
        (high_pass, low_pass), tr = bandpass
        logger.info('%s band-pass filtering (high-pass=%s, '
                    'low-pass=%s, tr=%s)'
                    % (time_series_file, high_pass, low_pass, tr))
        seed_series = fft_filter(seed_series, low_pass, high_pass, tr)
        target_series = fft_filter(target_series, low_pass, high_pass, tr)

    # Compute connectivity matrix
    r = seed_based_correlation(seed_series, target_series, True)

    # Set values slightly below 1 or above -1 (for use with, e.g., arctanh)
    r[r >= 1] = np.nextafter(np.float32(1.), np.float32(-1))
    r[r <= -1] = np.nextafter(np.float32(-1.), np.float32(1))

    if arctanh_transform:
        logger.info('%s applying arctanh transform' % connectivity_file)
        r = np.arctanh(r)

    if pca_transform:
        logger.info('%s applying PCA transform (n_components=%s)'
                    % (connectivity_file, pca_transform))
        r = detrend(r, axis=1, type='constant')
        pca = PCA(n_components=pca_transform)
        r = pca.fit_transform(r)

    # Ensure float32
    r = r.astype(np.float32)

    # Save output
    if compress:
        np.savez_compressed(connectivity_file, connectivity=r)
    else:
        np.savez(connectivity_file, connectivity=r)


def connectivity_dmri(input: dict, output: dict, params: dict,
                      log: list) -> None:
    """ Compute a connectivity matrix from functional data. This
    method uses FSL's probtrackx2 function which must be accessible
    from the terminal.

    Use '>> probtrackx2 --help' to get more information about the
    input arguments. The optional cubic and pca transformations
    are applied, in the mentioned order, after the connectivity
    matrix has been extracted from the FSL fdt_matrix2 output.

    Parameters
    ----------
    input : dict
        Input files, allowed: {fdt_matrix2, seed}
    output : dict
        Output file, allowed {connectivity}
    params : dict
        Parameters, allowed {cubic_transform, pca_transform, compress,
        cleanup_fsl}. For more information, see the CBPtools documentation on
        readthedocs.io under the parameters section for 'probtract_proc'.
    log : list
        Log files
    """

    fdt_matrix2_file = input.get('fdt_matrix2')
    seed_img_file = input.get('seed_mask')
    connectivity_file = output.get('connectivity')
    log_file = log[0]
    cubic_transform = params.get('cubic_transform', False)
    pca_transform = params.get('pca_transform', False)
    compress = params.get('compress', False)
    cleanup_fsl = params.get('cleanup_fsl', False)

    # Set up logging
    logger = get_logger('connectivity_dmri', log_file)

    i, j, value = np.loadtxt(fdt_matrix2_file, unpack=True)
    i = i.astype(int) - 1  # convert to int for indexing
    j = j.astype(int) - 1  # FSL indexes from 1, but we need 0-indexing

    r = coo_matrix((value, (i, j)))
    r = r.todense(order='F')

    if cubic_transform:
        logger.info('%s applying cubic transform' % connectivity_file)
        r = np.power(r, 1 / 3)

    if pca_transform:
        logger.info('%s applying PCA transform (n_components=%s)'
                    % (connectivity_file, pca_transform))
        r = detrend(r, axis=1, type='constant')
        pca = PCA(n_components=pca_transform)
        r = pca.fit_transform(r)

    # Reorder seed-voxels from F- to C-order
    seed = nib.load(seed_img_file)
    reorder = get_f2c_order(seed)
    r = r[reorder, :]

    # Ensure float32
    r = r.astype(np.float32)

    if compress:
        np.savez_compressed(connectivity_file, connectivity=r)
    else:
        np.savez(connectivity_file, connectivity=r)

    if cleanup_fsl:
        fsl_output = os.path.dirname(fdt_matrix2_file)
        logger.info('removing FSL\'s probtrackx2 output: %s' % fsl_output)
        rmtree(fsl_output)


def merge_sessions(input: dict, output: dict, params: dict, log: list) -> None:
    """ Merge multi-session connectivity matrices.

    If CBPtools receives multi-session input data, each session will have
    a corresponding connectivity matrix. This rule will merge all connectivity
    matrices over all sessions per subject.

    Parameters
    ----------
    input : dict
        Input files, allowed: {sessions}
    output : dict
        Output file, allowed {connectivity}
    params : dict
        Parameters, allowed {compress}
    log : list
        Log files
    """

    sessions = input.get('sessions')
    connectivity_file = output.get('connectivity')
    log_file = log[0]
    compress = params.get('compress')
    pca_transform = params.get('pca_transform', False)

    # Set up logging
    logger = get_logger('merge_sessions', log_file)

    r = []
    for session in sessions:
        _, ext = os.path.splitext(session)
        data = np.load(session)

        if ext == '.npz':
            data = data.get('connectivity')

        r.append(data)

    # Check if all matrices are filled
    for matrix in r:
        if matrix.size == 0:
            # At least one of the sessions has faulty data
            logger.error('could not create %s, at least one of the sessions'
                         'contains faulty data' % connectivity_file)
            np.savez(connectivity_file, connectivity=np.array([]))
            return

    # Merging strategy
    r = np.mean(r, axis=0)

    # Transforms
    if pca_transform:
        logger.info('%s applying PCA transform (n_components=%s)'
                    % (connectivity_file, pca_transform))
        r = detrend(r, axis=1, type='constant')
        pca = PCA(n_components=pca_transform)
        r = pca.fit_transform(r)

    if compress:
        np.savez_compressed(connectivity_file, connectivity=r)
    else:
        np.savez(connectivity_file, connectivity=r)
