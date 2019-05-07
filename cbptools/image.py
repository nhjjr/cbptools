#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for processing neuroimaging data.
Expected input to these functions are np.ndarrays rather than nibabel spatial images. This cuts down in computation
cost for loading data and formatting the data back into spatial image objects."""

from .exceptions import ShapeError
from sys import float_info
from scipy.ndimage.morphology import binary_dilation
from scipy.spatial import distance
from typing import Union, List
import nibabel as nib
import numpy as np

spatialimage = nib.spatialimages.SpatialImage


def imgs_equal_3d(imgs: List[spatialimage]) -> bool:
    """Checks whether the first 3 dimensions of input spatial images have the same shape and affine"""

    def _check_equal(iterator):
        iterator = iter(iterator)
        try:
            first = next(iterator)

        except StopIteration:
            return True

        return all(first == rest for rest in iterator)

    affines = [img.affine.flatten().tolist() for img in imgs]
    shapes = [img.shape[0:3] for img in imgs]

    return False if not _check_equal(affines) or not _check_equal(shapes) else True


def img_is_3d(img: spatialimage) -> bool:
    """ Check if the input spatial image is 3D """
    return False if len(img.shape) != 3 else True


def img_is_4d(img: spatialimage) -> bool:
    """ Check if the input spatial image is 3D """
    return False if len(img.shape) != 4 else True


def map_voxels(voxel_size: Union[list, np.ndarray], origin: Union[list, np.ndarray], shape: tuple) -> tuple:
    if len(voxel_size) == 1:
        voxel_size = np.repeat(voxel_size, 3)

    return shape, np.r_[np.c_[np.diag(np.array(np.abs(voxel_size)) * np.array([-1, 1, 1])), origin], [[0, 0, 0, 1]]]


def img_is_mask(img: np.ndarray, allow_empty: bool = True) -> bool:
    """Check if input array meets the criteria for being a mask (3D, binary, not empty)"""
    if not img_is_3d(img):
        return False

    elif not ((img == 0) | (img == 1)).all() and img.dtype is not bool:
        return False

    elif np.sum(img) == 0 and not allow_empty:
        return False

    return True


def binarize_3d(img: spatialimage, threshold: float = 0.0) -> spatialimage:
    """binarize 3D spatial image"""
    if not img_is_3d(img):
        raise ShapeError(3, len(img.shape))

    return nib.Nifti1Image(np.where(img.get_data() > threshold, 1, 0), img.affine, img.header)


def subtract_img(source_img: spatialimage, target_img: spatialimage, edge_dist: int = 0) -> spatialimage:
    """Subtracts 3D array y from x. Optionally, y is expanded by the edge_dist value (in milimeters), using the x and y
    affine values to calculate Euclidean distance."""

    source_data = source_img.get_data()  # min
    target_data = target_img.get_data()  # sub

    if source_data.shape != target_data.shape:
        raise ShapeError(source_data.shape, target_data.shape)

    difference = np.zeros(source_data.shape)
    source_voxels = np.asarray(np.where(source_data == 1)).transpose()
    target_voxels = np.asarray(np.where(target_data == 1)).transpose()

    # Calculate Euclidean distance from outermost voxels in y
    source_ref = nib.affines.apply_affine(source_img.affine, source_voxels)
    target_ref = nib.affines.apply_affine(target_img.affine, target_voxels)
    dist = distance.cdist(target_ref, source_ref, 'euclidean')
    dist = np.amin(dist, axis=0)  # for each target voxel get the minimum distance value
    x, y, z = source_voxels[np.where(dist > edge_dist), :].squeeze().transpose()
    difference[x, y, z] = 1

    return nib.Nifti1Image(np.float32(difference), source_img.affine, source_img.header)


def subsample_img(img: spatialimage, f: int = 2) -> spatialimage:
    """Reduce image features of a 3D array by a given factor f."""
    if not img_is_3d(img):
        raise ShapeError(3, len(img.shape))

    data = img.get_data()
    mask = np.zeros(img.shape)
    mask[::f, ::f, ::f] = 1
    data *= mask
    return nib.Nifti1Image(data, img.affine, img.header)


def median_filter_img(img: spatialimage, dist: int = 1) -> spatialimage:
    """Median filtering of non-zero elements in an image.

    Median filter all selected voxels of a binary 3D input image and a 2-iteration dilation border around it. Voxel
    distance dist is used to determine the size of the area for computing the median, where a distance of 1 results in
    a 3*3*3 shape. The number of repetitions nrep determines how often the median filter will be repeated.
    """

    if not img_is_3d(img):
        raise ShapeError(3, len(img.shape))

    data = img.get_data()
    dilated = binary_dilation(data, iterations=2).astype(int)
    voxels = np.asarray(np.where(dilated == 1)).transpose()

    filtered = np.zeros(img.shape)
    for x, y, z in voxels:
        area = data[x - dist:x + (dist + 1), y - dist:y + (dist + 1), z - dist:z + (dist + 1)]

        if np.median(area) > float_info.min:
            filtered[x, y, z] = 1

    return nib.Nifti1Image(np.float32(filtered), img.affine, img.header)


def stretch_img(source_img: spatialimage, target: Union[tuple, spatialimage]) -> spatialimage:
    """Stretch a binary image to meet the dimensions of a template image.

    The stretching process will not generate new indices of ones, instead keeping the original amount of mask values
    but spacing them out over the template dimensions. This function assumes that the template dimensions are larger
    than the input image dimensions and the input image is binary.
    """

    try:
        target_shape, target_affine = target.shape, target.affine

    except AttributeError:
        target_shape, target_affine = target

    if np.all(np.abs(np.diag(source_img.affine[:3])) <= np.abs(np.diag(target_affine[:3]))):
        return ValueError('This function is meant for upsampling, not downsampling')

    x, y, z = np.nonzero(source_img.get_data())
    xyz = np.asarray([x, y, z, np.ones(len(z))]).T

    # 'Stretch' coordinates so they space out in a larger template
    xyz = np.unique(np.round(np.diag(np.linalg.solve(target_affine, source_img.affine)) * xyz), axis=0).astype(int)
    x, y, z, _ = np.hsplit(xyz, 4)
    m = np.zeros(target_shape)
    m[x, y, z] = 1

    return nib.Nifti1Image(np.float32(m), target_affine, source_img.header)


def get_masked_series(time_series: spatialimage, mask: spatialimage):
    """Apply a 3D mask to a 4D image

    Parameters
    ----------
    time_series : spatialimage
        4D nifti image, with time-series on the 4th dimension
    mask : spatialimage
        3D boolean mask image
    """
    if not imgs_equal_3d([time_series, mask]):
        raise ValueError('Time-series and mask do not have equal shape and/or affine')

    if not img_is_4d(time_series):
        raise ShapeError(4, len(time_series.shape))

    time_series_data = time_series.get_data()
    mask_data = mask.get_data().astype(bool)

    return time_series_data[mask_data].T


def find_low_variance_voxels(data, tol: float = np.finfo(np.float32).eps):
    return np.where(data.var(axis=0) < tol)[0]


def get_mask_indices(img: spatialimage, order: str = 'C') -> np.ndarray:
    """Get voxel space coordinates (indices) of seed voxels

    Parameters
    ----------
    img : spatialimage
        Mask NIfTI image
    order : str, optional
        Order that the seed-mask voxels will be extracted in. The resulting indices will be listed in this way

    Returns
    -------
    np.ndarray
        2D array of shape (n_voxels, 3) containing the 3D coordinates of all mask image voxels
    """
    data = img.get_data()
    indices = np.asarray(tuple(zip(*np.where(img.get_data() == 1))))

    if order != 'C':
        reorder = np.argsort(np.arange(np.prod(img.shape)).reshape(img.shape, order=order)[data.astype(bool)])
        indices = indices[reorder]

    return indices


def map_labels(img: spatialimage, labels: np.ndarray, indices: np.ndarray) -> spatialimage:
    """Map cluster labels onto the seed mask

    Parameters
    ----------
    img : spatialimage
        Mask NIfTI image to which the labels will be mapped
    labels : np.ndarray
        1D array of cluster labels (sklearn.cluster.KMeans._labels)
    indices : np.ndarray
        Indices of all mask image voxels of which the order coincides with
        the order of the voxels in the labels array.

    Returns
    -------
    spatialimage
        Mask image with the labels mapped onto it.
    """
    if len(indices) != len(labels):
        raise ValueError('Indices and labels do not match')

    mapped_img = np.zeros(img.shape)
    mapped_img[indices[0:, 0], indices[0:, 1], indices[0:, 2]] = labels
    return nib.Nifti1Image(np.float32(mapped_img), img.affine, img.header)
