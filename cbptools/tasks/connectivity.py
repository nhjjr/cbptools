#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cbptools.exceptions import DimensionError
from cbptools.image import (img_is_4d, get_masked_series,
                            find_low_variance_voxels, get_f2c_order)
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


def connectivity_rsfmri(input: dict, output: dict, params: dict) -> None:
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
        Output files, allowed {connectivity, log_file}
    params : dict
        Parameters, allowed {participant_id, arctanh_transform, pca_transform,
        bandpass, low_variance_error, compress, confounds, smoothing_fwhm}
        For more information, see the CBPtools documentation on readthedocs.io
        under the parameters section for 'time_series_proc'.
    """

    # input, output, params
    time_series_file = input.get('time_series')
    seed_img_file = input.get('seed_mask')
    target_img_file = input.get('target_mask')
    confounds_file = input.get('confounds', None)
    log_file = output.get('log_file')
    connectivity_file = output.get('connectivity')
    participant_id = params.get('participant_id')
    session_id = params.get('session_id', None)
    smoothing = params.get('smoothing', False)
    lv_correction = params.get('low_variance_correction', False)
    lv_in_seed = params.get('low_variance_in_seed', None)
    lv_in_target = params.get('low_variance_in_target', None)
    lv_behavior = params.get('low_variance_behavior', None)
    confounds_sep = params.get('confounds_delimiter', None)
    confounds_cols = params.get('confounds_columns', None)
    bandpass = params.get('bandpass', False)
    pca_transform = params.get('pca_transform', False)
    arctanh_transform = params.get('arctanh_transform', False)
    compress = params.get('compress', False)

    # Load input data
    time_series = nib.load(time_series_file)
    seed_img = nib.load(seed_img_file)
    target_img = nib.load(target_img_file)

    if not img_is_4d(time_series):
        raise DimensionError(4, len(time_series.shape))

    if smoothing:
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

    if session_id:
        # If a session_id is set, then include it in the log file
        data = [[participant_id, session_id, in_seed, in_target,
                 bad_seed or bad_target]]
        columns = ['participant_id', 'session_id', 'low_variance_in_seed',
                   'low_variance_in_target', 'low_variance_excluded']
    else:
        data = [[participant_id, in_seed, in_target, bad_seed or bad_target]]
        columns = ['participant_id', 'low_variance_in_seed',
                   'low_variance_in_target', 'low_variance_excluded']

    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(log_file, sep='\t', index=False)
    del df

    # If the participant has data exceeding the seed- or target low
    # variance threshold, output an empty file
    if lv_correction:
        if bad_seed or bad_target:
            np.savez(connectivity_file, connectivity=np.array([]))
            return

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

        confounds = confounds.values
        seed_series = nuisance_signal_regression(
            seed_series, confounds=confounds, demean=False)
        target_series = nuisance_signal_regression(
            target_series, confounds=confounds, demean=False)

    # Apply band-pass filter if high_pass, low_pass, and tr are defined
    if bandpass:
        (high_pass, low_pass), tr = bandpass
        seed_series = fft_filter(seed_series, low_pass, high_pass, tr)
        target_series = fft_filter(target_series, low_pass, high_pass, tr)

    # Compute connectivity matrix
    r = seed_based_correlation(seed_series, target_series, True)

    # Set values slightly below 1 or above -1 (for use with, e.g., arctanh)
    r[r >= 1] = np.nextafter(np.float32(1.), np.float32(-1))
    r[r <= -1] = np.nextafter(np.float32(-1.), np.float32(1))

    if arctanh_transform:
        r = np.arctanh(r)

    if pca_transform:
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


def merge_connectivity_logs(input: dict, output: dict) -> None:
    """Merge all connectivity output logs"""
    # input, output
    log_file = input.get('log')
    merged_log_file = output.get('merged_log')

    df = pd.concat((
        pd.read_csv(f, sep='\t', index_col=False)
        for f in log_file), axis=0).reset_index(drop=True)

    df.to_csv(merged_log_file, sep='\t', index=False)


def validate_connectivity(input: dict, output: dict, params: dict) -> None:
    """Ensure that all connectivity matrices could be computed"""

    # input, output, params
    log_file = input.get('log')
    touch_file = output.get('touchfile')
    connectivity_template = params.get('connectivity')
    labels_template = params.get('labels')
    n_clusters = params.get('n_clusters')

    df = pd.read_csv(log_file, sep='\t')
    bad_participants = df[df['low_variance_excluded']]['participant_id']
    bad_participants = list(np.unique(bad_participants.tolist()))

    # Remove files with erroneous or no data
    if bad_participants:
        print('--------------------------------------------------------------')
        print('%s subject(s) with problematic data.' % len(bad_participants))
        print('--------------------------------------------------------------')

        for bad_participant in bad_participants:
            connectivity_file = connectivity_template.format(
                participant_id=bad_participant
            )
            labels_files = [labels_template.format(
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

        print('\nOpen %s to find problematic participants in the '
              'low_variance_excluded column\n' % log_file)
        raise ValueError('Participants with bad data found. Open '
                         '%s for more details.' % log_file)

    # Touch an output file that subsequent rules depend on
    else:
        Path(touch_file).touch()


def connectivity_dmri(input: dict, output: dict, params: dict) -> None:
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
    """

    fdt_matrix2_file = input.get('fdt_matrix2')
    seed_img_file = input.get('seed_mask')
    connectivity_file = output.get('connectivity')
    cubic_transform = params.get('cubic_transform', False)
    pca_transform = params.get('pca_transform', False)
    compress = params.get('compress', False)
    cleanup_fsl = params.get('cleanup_fsl', False)

    i, j, value = np.loadtxt(fdt_matrix2_file, unpack=True)
    i = i.astype(int) - 1  # convert to int for indexing
    j = j.astype(int) - 1  # FSL indexes from 1, but we need 0-indexing

    r = coo_matrix((value, (i, j)))
    r = r.todense(order='F')

    if cubic_transform:
        r = np.power(r, 1 / 3)

    if pca_transform:
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
        rmtree(os.path.dirname(fdt_matrix2_file))


def merge_sessions(input: dict, output: dict, params: dict) -> None:
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
    """

    sessions = input.get('sessions')
    connectivity_file = output.get('connectivity')
    compress = params.get('compress')
    arctanh_transform = params.get('arctanh_transform', False)
    pca_transform = params.get('pca_transform', False)
    cubic_transform = params.get('cubic_transform', False)

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
            np.savez(connectivity_file, connectivity=np.array([]))
            return

    # Merging strategy
    r = np.mean(r, axis=0)

    # Transforms
    r[r >= 1] = np.nextafter(np.float32(1.), np.float32(-1))
    r[r <= -1] = np.nextafter(np.float32(-1.), np.float32(1))

    if arctanh_transform:
        r = np.arctanh(r)

    if cubic_transform:
        r = np.power(r, 1 / 3)

    if pca_transform:
        r = detrend(r, axis=1, type='constant')
        pca = PCA(n_components=pca_transform)
        r = pca.fit_transform(r)

    if compress:
        np.savez_compressed(connectivity_file, connectivity=r)
    else:
        np.savez(connectivity_file, connectivity=r)
