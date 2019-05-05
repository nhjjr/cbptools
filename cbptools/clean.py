#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data cleaning utilities"""
from .exceptions import ShapeError
import numpy as np


def fft_filter(x: np.ndarray, low_pass: float, high_pass: float, tr: float) -> np.ndarray:
    """ Uses Fast-Fourier Transform to remove signals outside of the defined frequency band.

    Parameters
    ----------
    x : np.ndarray
        timepoints by voxels time series
    low_pass : float
        Low-pass filter value
    high_pass : float
        High-pass filter value
    tr : float
        fMRI repetition time in seconds

    Returns
    -------
    np.ndarray
        Filtered time series
    """
    if high_pass >= low_pass:
        raise ValueError(f'High pass ({high_pass}) should be smaller than low pass ({low_pass})')

    f = np.fft.fftfreq(len(x), d=tr)
    idx = np.where((abs(f) < high_pass) | (abs(f) > low_pass))[0]
    idx = idx[idx > 0]  # Remove intercept
    x = np.fft.fft(x, axis=0)
    x[idx, :] = 0
    x = np.real(np.fft.ifft(x, axis=0)).astype(np.float32)
    return x


def nuisance_signal_regression(data: np.ndarray, confounds: np.ndarray, demean: bool = False) -> np.ndarray:
    """Nuisance signal regression of confounds on data."""

    if data.shape[0] != confounds.shape[0]:
        raise ShapeError(data.shape[0], confounds.shape[0])

    if demean is True:
        if np.all(np.round(confounds.mean(axis=0), decimals=10) == 0) is not True:
            confounds = confounds - confounds.mean(axis=0)

    data = data - np.dot(confounds, np.linalg.lstsq(confounds, data, rcond=-1)[0])
    return data.astype(np.float32)

