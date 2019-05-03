#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cbptools.exceptions import DimensionError
from cbptools.image import img_is_4d, apply_mask
from cbptools.connectivity import seed_based_correlation
from cbptools.clean import nuisance_signal_regression, fft_filter
from cbptools import FSL
from scipy.signal import detrend
from sklearn.decomposition import PCA
from nibabel.processing import smooth_image
from fnmatch import fnmatch
import pandas as pd
import nibabel as nib
import numpy as np
import gc
import os


def fmri_conn(time_series: str, seed: str, target: str, participant_id: str, file: str, log_file: str,
              seed_low_variance: float=0.05, target_low_variance: float=0.1, smoothing_fwhm: int=None,
              confounds: dict=None, band_pass: tuple=None, arctanh_transform: bool=True,
              pca_transform: float=None) -> None:
    """ Compute a connectivity matrix from functional data.

    Processing steps:
      [1] Apply smoothing to time-series data (optional),
      [2] Apply ROI and target mask to the time-series,
      [3] apply nuisance signal regression on the roi- and target-masked time-series separately (optional),
      [4] apply band-pass filtering on the roi- and target-masked time-series separately (optional),
      [5] compute the correlation between the roi-masked time-series and target-masked time-series,
      [6] apply an arctanh transformation (np.arctanh) on the connectivity matrix (optional),
      [7] apply principal component analysis on the connectivity matrix (optional)

    Parameters
    ----------
    time_series: str
        Path to the time-series nifti image
    seed: str
        Path to the region-of-interest mask nifti image
    target: str
        Path to the (whole-brain) target mask nifti image
    participant_id: str
        Unique identifier of the participant currently being processed
    file: str
        Output filename for the connectivity matrix (.npy)
    log_file : str
        Output filename for the log file (.log)
    seed_low_variance : float
        Percentage of low-variance voxels (tolerance of np.finfo(np.float32).eps) allowed to be within the seed
    target_low_variance : float
        Percentage of low-variance voxels (tolerance of np.finfo(np.float32).eps) allowed to be within the target
    smoothing_fwhm: int, optional
        Smoothing kernel at FWHM in milimeters. If None, smoothing is skipped.
    confounds: dict, optional
        Confounds file and reading parameters for nuisance signal regression:
          'file': str, path to the confounds file (must be a tabular plain-text file with a header)
          'sep': str, separator used (e.g., .tsv has '\t', .csv has ',' or ';')
          'usecols': list, containing all columns to be used. If this is not given, all columns are used.
    band_pass: tuple, optional
        Band-pass filter parameters. The following order should be maintained for the tuple: (high_pass, low_pass, tr).
        If one is not given, band-pass filtering cannot proceed.
    arctanh_transform: bool, optional
        Apply np.arctanh to the connectivity matrix. This will be applied by default.
    pca_transform: float, optional
        Apply sklearn.decomposition.PCA to the connectivity matrix. The parameter value is used as the n_components
        value specified in the description of the aforementioned class. All other parameters are kept to sklearn
        defaults. Applying this returns the ROI-voxels by principal components as a connectivity matrix. If this is set
        to None, this step will be ignored.
    """

    def _find_low_variance_voxels(data, tol: float = np.finfo(np.float32).eps):
        return np.where(data.var(axis=0) < tol)[0]

    time_series = nib.load(time_series)
    seed_img = nib.load(seed)
    target_img = nib.load(target)

    if not img_is_4d(time_series):
        raise DimensionError(4, len(time_series.shape))

    if smoothing_fwhm is not None:
        time_series = smooth_image(time_series, fwhm=smoothing_fwhm)

    seed_series = apply_mask(time_series, mask_img=seed_img, as_array=True)
    target_series = apply_mask(time_series, mask_img=target_img, as_array=True)
    del time_series
    gc.collect()

    # Identify low-variance voxels and log them
    in_seed = _find_low_variance_voxels(data=seed_series)
    in_target = _find_low_variance_voxels(data=target_series)
    bad_seed = in_seed.size/np.sum(seed_img.get_data() > 0) > seed_low_variance
    bad_target = in_target.size/np.sum(target_img.get_data() > 0) > target_low_variance
    pd.DataFrame(
        data=[[participant_id, in_seed, in_target, bad_seed or bad_target]],
        columns=['participant_id', 'low_variance_in_seed', 'low_variance_in_target', 'low_variance_excluded']
    ).to_csv(log_file, sep='\t', index=False)

    if bad_seed or bad_target:
        raise ValueError(f'{participant_id}: Too many low-variance voxels in seed ({len(in_seed)}) or target '
                         f'({len(in_target)})')

    if confounds is not None:  # Nuisance Signal Regression
        confounds_file = confounds.get('file')
        sep = confounds.get('sep')
        usecols = set(confounds.get('usecols'))

        # Check if usecols contains wildcards to extend upon the header
        if usecols is not None:
            header = pd.read_csv(confounds_file, sep=sep, header=None, nrows=1).values.tolist()[0]
            usecols = [x for x in header if any(fnmatch(x, p) for p in usecols)]

        confounds = pd.read_csv(confounds_file, sep=sep, usecols=usecols).values
        seed_series = nuisance_signal_regression(seed_series, confounds=confounds, demean=False)
        target_series = nuisance_signal_regression(target_series, confounds=confounds, demean=False)

    high_pass, low_pass, tr = band_pass
    if all([low_pass, high_pass, tr]):  # apply band-pass filter
        seed_series = fft_filter(seed_series, low_pass=low_pass, high_pass=high_pass, tr=tr)
        target_series = fft_filter(target_series, low_pass=low_pass, high_pass=high_pass, tr=tr)

    connectivity = seed_based_correlation(x=seed_series, y=target_series, standardize=True)

    if arctanh_transform:
        connectivity = np.arctanh(connectivity)

    if pca_transform is not None:
        connectivity = detrend(connectivity, axis=1, type='constant')  # zero-mean columns (target)
        pca = PCA(n_components=pca_transform)
        connectivity = pca.fit_transform(connectivity)

    np.save(file, connectivity)


def dfmri_conn(seed: str, target: str, samples: str, bet_binary_mask: str, tmp_dir: str, xfm: str, inv_xfm: str,
               file: str, pd: bool=True, n_samples: int=200, n_steps: int=2000, step_length: float=0.5,
               dist_thresh: float=5.0, c_thresh: float=0.2, loop_check: bool=True, cubic_transform: bool=True,
               pca_transform: bool=False, wait_for_file: int=240, cleanup_fsl: bool=True) -> None:
    """ Compute a connectivity matrix from functional data. This method uses FSL's probtrackx2 function which must be
    accessible from the terminal.

    Use '>> probtrackx2 --help' to get more information about the input arguments. The optional cubic and pca
    transformations are applied, in the mentioned order, after the connectivity matrix has been extracted from the
    FSL fdt_matrix2 output.
    
    Parameters
    ----------
    seed : str
        Path to the region-of-interest high-resolution mask nifti image. Used for the -x,--seed argument.
    target : str
        Path to the low-resolution target mask nifti image. Used for the --target2 argument.
    samples : str
        Basename for samples files (e.g., 'merged'). Used for the -s,--samples argument.
    bet_binary_mask : str
        Bet binary mask file in diffusion space. Used for the -m,--mask argument.
    tmp_dir : str
        Directory to put the FSL final volume output in. Used for the --dir argument. If cleanup_fsl is set to True,
        this directory will be deleted after the connectivity matrix is extracted.
    xfm : str
        Transform taking seed space to DTI space (either FLIRT matrix of FNIRT warpfield). Used for the --xfm argment.
    inv_xfm : str
        Transform taking DTI space to seed space. Used for the --invxfm argument.
    file : str
        Output filename for the connectivity matrix (.npy)
    pd : bool, optional
        Correct path distribution for the length of the pathways. Used for the --pd argument.
    n_samples : int, optional
        Number of samples, default is 200. Used for the -P,--nsamples argument.
    n_steps : int, optional
        Number of steps per sample, default is 2000. Used for the -S,--nsteps argument.
    step_length : float, optional
        Steplength in mm, default is 0.5. Used for the --steplength argument.
    dist_thresh : float, optional
        Discards samples shorter than this threshold in mm, default is 5.0. Used for the --distthresh argument.
    c_thresh : float, optional
        Curvature threshold, default is 0.2. Used for the -c,--cthr argument.
    loop_check: bool, optional
        Perform loopchecks on paths, default is True. Used for the -l,--loopcheck argument.
    cubic_transform : bool, optional
        Apply a cubic transformation to the connectivity matrix.
    pca_transform : float, optional
        Apply sklearn.decomposition.PCA to the connectivity matrix. The parameter value is used as the n_components
        value specified in the description of the aforementioned class. All other parameters are kept to sklearn
        defaults. Applying this returns the ROI-voxels by principal components as a connectivity matrix. If this is set
        to None, this step will be ignored.
    wait_for_file: int, optional
        Wait for the FSL output file 'fdt_matrix2' to appear in the file system in seconds, default is 240. If the file
        has not appeared within this time, the script assumes something went wrong.
    cleanup_fsl: bool, optional
        Remove the FSL output directory defined in tmp_dir. Once the connectivity matrix has been extracted, this data
        will no longer be used by the pipeline.
    """

    options = (f'--nsamples={n_samples}', f'--nsteps={n_steps}', f'--steplength={step_length}',
               f'--distthresh={dist_thresh}', f'--cthr={c_thresh}', '--omatrix2', '--forcedir', '--verbose=0')

    if loop_check:
        options += ('-l',)

    if pd:
        options += ('--pd',)

    fsl = FSL()
    connectivity = fsl.run_probtrackx2(
        seed=seed,
        target=target,
        samples=samples,
        mask=bet_binary_mask,
        tmp_dir=tmp_dir,
        xfm=xfm,
        invxfm=inv_xfm,
        options=options,
        wait_for_file=wait_for_file,
        cleanup_fsl=cleanup_fsl
    )

    if cubic_transform:
        connectivity = np.power(connectivity, 1 / 3)

    if pca_transform is not None:
        connectivity = detrend(connectivity, axis=1, type='constant')  # zero-mean columns (target)
        pca = PCA(n_components=pca_transform)
        connectivity = pca.fit_transform(connectivity)

    np.save(file, connectivity)


if __name__ == '__main__':
    modality = snakemake.params.get('modality')

    if modality == 'fmri':
        # Format the confounds file input
        confounds_file = snakemake.params.get('confounds_file', None)
        confounds = {'file': None, 'sep': None, 'usecols': None}
        if isinstance(confounds_file, list):
            for value in confounds_file[1:]:
                if value.startswith('sep=') or value.startswith('delimiter='):
                    confounds_sep = value.split(sep='=')[-1]
                    break
            else:
                confounds_sep = None

            confounds = {
                'file': confounds_file[0].format(participant_id=snakemake.wildcards.get('participant_id')),
                'sep': confounds_sep,
                'usecols': snakemake.params.get('confounds_usecols', None)
            }
        elif confounds is not None:
            confounds = {
                'file': confounds_file.format(participant_id=snakemake.wildcards.get('participant_id')),
                'sep': None,
                'usecols': snakemake.params.get('confounds_usecols', None)
            }

        fmri_conn(
            time_series=snakemake.input.get('time_series'),
            seed=snakemake.input.get('seed'),
            target=snakemake.input.get('target'),
            participant_id=snakemake.wildcards.get('participant_id'),
            file=snakemake.output[0],
            log_file=snakemake.params.get('log'),
            seed_low_variance=snakemake.params.get('seed_low_variance', 0.05),
            target_low_variance=snakemake.params.get('target_low_variance', 0.1),
            smoothing_fwhm=snakemake.params.get('smoothing_fwhm', None),
            confounds=confounds,
            band_pass=(
                snakemake.params.get('high_pass', None),
                snakemake.params.get('low_pass', None),
                snakemake.params.get('tr', None)
            ),
            arctanh_transform=snakemake.params.get('arctanh_transform', False),
            pca_transform=snakemake.params.get('pca_transform', None)
        )

    elif modality == 'dmri':
        dfmri_conn(
            seed=snakemake.input.get('seed'),
            target=snakemake.input.get('target'),
            samples=snakemake.params.get('samples'),
            bet_binary_mask=snakemake.input.get('bet_binary_mask'),
            tmp_dir=snakemake.params.get('tmp_dir'),
            xfm=snakemake.input.get('xfm'),
            inv_xfm=snakemake.input.get('inv_xfm'),
            file=snakemake.output[0],
            pd=snakemake.params.get('pd'),
            n_samples=snakemake.params.get('n_samples'),
            n_steps=snakemake.params.get('n_steps'),
            step_length=snakemake.params.get('step_length'),
            dist_thresh=snakemake.params.get('dist_thresh'),
            c_thresh=snakemake.params.get('c_thresh'),
            loop_check=snakemake.params.get('loop_check'),
            cubic_transform=snakemake.params.get('cubic_transform'),
            pca_transform=snakemake.params.get('pca_transform'),
            wait_for_file=snakemake.params.get('wait_for_file'),
            cleanup_fsl=snakemake.params.get('cleanup_fsl')
        )

    else:
        raise ValueError(f'Modality not recognized: Expected fmri or dmri, got {modality}')
