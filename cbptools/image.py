#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for processing neuroimaging data.
Expected input to these functions are np.ndarrays rather than nibabel
spatial images. This cuts down in computation cost for loading data
and formatting the data back into spatial image objects."""

from .exceptions import ShapeError
from scipy.ndimage.morphology import binary_dilation
from nibabel.spatialimages import SpatialImage
from scipy.spatial import distance
from typing import Union, List
from sys import float_info
import nibabel as nib
import numpy as np
import pkg_resources


def imgs_equal_3d(imgs: List[SpatialImage]) -> bool:
    """Checks whether the first 3 dimensions of input spatial
    images have the same shape and affine"""

    def _check_equal(iterator):
        iterator = iter(iterator)
        try:
            first = next(iterator)

        except StopIteration:
            return True

        return all(first == rest for rest in iterator)

    affines = [img.affine.flatten().tolist() for img in imgs]
    shapes = [img.shape[0:3] for img in imgs]

    return False if not _check_equal(affines) or not _check_equal(shapes) \
        else True


def img_is_3d(img: SpatialImage) -> bool:
    """ Check if the input spatial image is 3D """
    return False if len(img.shape) != 3 else True


def img_is_4d(img: SpatialImage) -> bool:
    """ Check if the input spatial image is 3D """
    return False if len(img.shape) != 4 else True


def map_voxels(voxel_size: Union[list, np.ndarray],
               origin: Union[list, np.ndarray], shape: tuple) -> tuple:
    if len(voxel_size) == 1:
        voxel_size = np.repeat(voxel_size, 3)

    mapping = np.r_[
        np.c_[
            np.diag(np.array(np.abs(voxel_size)) * np.array([-1, 1, 1])),
            origin
        ], [[0, 0, 0, 1]]
    ]

    return shape, mapping


def img_is_mask(img: np.ndarray, allow_empty: bool = True) -> bool:
    """Check if input array meets the criteria for being a mask
    (3D, binary, not empty)"""
    if not img_is_3d(img):
        return False

    elif not ((img == 0) | (img == 1)).all() and img.dtype is not bool:
        return False

    elif np.sum(img) == 0 and not allow_empty:
        return False

    return True


def binarize_3d(img: SpatialImage, threshold: float = 0.0) -> SpatialImage:
    """binarize 3D spatial image. NaNs and infs in the image are set to 0."""
    if not img_is_3d(img):
        raise ShapeError(3, len(img.shape))

    data = img.get_data()
    data[np.where(np.isnan(data))] = 0
    data[np.where(np.isinf(data))] = 0
    data = np.where(data > threshold, 1, 0)

    return nib.Nifti1Image(data, img.affine, img.header)


def subtract_img(source_img: SpatialImage, target_img: SpatialImage,
                 edge_dist: int = 0) -> SpatialImage:
    """Subtracts 3D array y from x. Optionally, y is expanded by the
    edge_dist value (in milimeters), using the x and y affine values
    to calculate Euclidean distance."""

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

    # for each target voxel get the minimum distance value
    dist = np.amin(dist, axis=0)
    x, y, z = source_voxels[np.where(dist > edge_dist), :].squeeze()\
        .transpose()
    difference[x, y, z] = 1

    return nib.Nifti1Image(np.float32(difference), source_img.affine,
                           source_img.header)


def subsample_img(img: SpatialImage, f: int = 2) -> SpatialImage:
    """Reduce image features of a 3D array by a given factor f."""
    if not img_is_3d(img):
        raise ShapeError(3, len(img.shape))

    data = img.get_data().astype(int)
    mask = np.zeros(img.shape).astype(int)
    mask[::f, ::f, ::f] = 1
    data *= mask
    return nib.Nifti1Image(data, img.affine, img.header)


def median_filter_img(img: SpatialImage, dist: int = 1) -> SpatialImage:
    """Median filtering of non-zero elements in an image.

    Median filter all selected voxels of a binary 3D input image
    and a 2-iteration dilation border around it. Voxel distance
    dist is used to determine the size of the area for computing
    the median, where a distance of 1 results in a 3*3*3 shape.
    The number of repetitions nrep determines how often the median
    filter will be repeated.
    """

    if not img_is_3d(img):
        raise ShapeError(3, len(img.shape))

    data = img.get_data()
    dilated = binary_dilation(data, iterations=2).astype(int)
    voxels = np.asarray(np.where(dilated == 1)).transpose()

    filtered = np.zeros(img.shape)
    for x, y, z in voxels:
        area = data[x-dist:x+(dist+1), y-dist:y+(dist+1), z-dist:z+(dist+1)]

        if np.median(area) > float_info.min:
            filtered[x, y, z] = 1

    return nib.Nifti1Image(np.float32(filtered), img.affine, img.header)


def stretch_img(source_img: SpatialImage,
                target: Union[tuple, SpatialImage]) -> SpatialImage:
    """Stretch a binary image to meet the dimensions of a
    template image.

    The stretching process will not generate new indices of ones,
    instead keeping the original amount of mask values but spacing
    them out over the template dimensions. This function assumes
    that the template dimensions are larger than the input image
    dimensions and the input image is binary.
    """

    try:
        target_shape, target_affine = target.shape, target.affine

    except AttributeError:
        target_shape, target_affine = target

    s_affine = np.abs(np.diag(source_img.affine[:3]))
    t_affine = np.abs(np.diag(target_affine[:3]))
    if np.all(s_affine <= t_affine):
        raise ValueError('This function is meant for upsampling, not '
                         'downsampling')

    x, y, z = np.nonzero(source_img.get_data())
    xyz = np.asarray([x, y, z, np.ones(len(z))]).T

    # 'Stretch' coordinates so they space out in a larger template
    xyz = np.diag(np.linalg.solve(target_affine, source_img.affine)) * xyz
    xyz = np.unique(np.round(xyz), axis=0)
    xyz = xyz.astype(int)
    x, y, z, _ = np.hsplit(xyz, 4)
    m = np.zeros(target_shape)
    m[x, y, z] = 1

    return nib.Nifti1Image(np.float32(m), target_affine, source_img.header)


def get_masked_series(time_series: SpatialImage, mask: SpatialImage):
    """Apply a 3D mask to a 4D image

    Parameters
    ----------
    time_series : SpatialImage
        4D nifti image, with time-series on the 4th dimension
    mask : SpatialImage
        3D boolean mask image
    """
    if not imgs_equal_3d([time_series, mask]):
        raise ValueError('Time-series and mask do not have equal shape '
                         'and/or affine')

    if not img_is_4d(time_series):
        raise ShapeError(4, len(time_series.shape))

    time_series_data = time_series.get_data()
    mask_data = mask.get_data().astype(bool)

    return time_series_data[mask_data].T


def find_low_variance_voxels(data, tol: float = np.finfo(np.float32).eps):
    return np.where(data.var(axis=0) < tol)[0]


def get_f2c_order(img: SpatialImage) -> np.ndarray:
    """The order in which voxels are extracted from a mask is either
    F-contiguous or C-contiguous (C by default in NumPy), which is reflected
    by the order in which they are placed in an array.

    This function provides reordering indices such that an F extraction order
    is turned into a C extraction order.

    Parameters
    ----------
    img : SpatialImage
        Mask NIfTI image
    """
    mask = img.get_data()
    reorder = np.arange(int(np.prod(img.shape)))
    reorder = reorder.reshape(img.shape, order='C')
    reorder = reorder.flatten(order='F')
    reorder = reorder[mask.flatten(order='F').astype(bool)]
    reorder = np.argsort(reorder)
    return reorder


def get_c2f_order(img: SpatialImage) -> np.ndarray:
    """The order in which voxels are extracted from a mask is either
    F-contiguous or C-contiguous (C by default in NumPy), which is reflected
    by the order in which they are placed in an array. 
    
    This function provides reordering indices such that a C extraction order 
    is turned into an F extraction order.

    Parameters
    ----------
    img : SpatialImage
        Mask NIfTI image
    """
    mask = img.get_data()
    reorder = np.arange(int(np.prod(mask.shape)))
    reorder = reorder.reshape(mask.shape, order='F')
    reorder = reorder[mask.astype(bool)]
    reorder = np.argsort(reorder)
    return reorder


def get_mask_indices(img: SpatialImage, order: str = 'C') -> np.ndarray:
    """Get voxel space coordinates (indices) of seed voxels

    Parameters
    ----------
    img : SpatialImage
        Mask NIfTI image
    order : str, optional
        Order that the seed-mask voxels will be extracted in.
        The resulting indices will be listed in this way

    Returns
    -------
    np.ndarray
        2D array of shape (n_voxels, 3) containing the 3D coordinates of
        all mask image voxels
    """

    if order not in ('C', 'F', 'c', 'f'):
        raise ValueError('Order has unexpected value: expected %s, got \'%s\''
                         % ("'C' or 'F'", order))

    data = img.get_data()
    indices = np.asarray(tuple(zip(*np.where(data == 1))))

    if order.upper() == 'F':
        # indices are C order and must become F order
        reorder = get_c2f_order(img)
        indices = indices[reorder]

    return indices


def map_labels(img: SpatialImage, labels: np.ndarray,
               indices: np.ndarray) -> SpatialImage:
    """Map cluster labels onto the seed mask

    Parameters
    ----------
    img : SpatialImage
        Mask NIfTI image to which the labels will be mapped
    labels : np.ndarray
        1D array of cluster labels (sklearn.cluster.KMeans._labels)
    indices : np.ndarray
        Indices of all mask image voxels of which the order coincides
        with the order of the voxels in the labels array.

    Returns
    -------
    SpatialImage
        Mask image with the labels mapped onto it.
    """
    if len(indices) != len(labels):
        raise ValueError('Indices and labels do not match')

    mapped_img = np.zeros(img.shape)
    mapped_img[indices[0:, 0], indices[0:, 1], indices[0:, 2]] = labels
    return nib.Nifti1Image(np.float32(mapped_img), img.affine, img.header)


def make_hollow(arr: np.ndarray) -> SpatialImage:
    """ Hollow out a volumetric image so that only voxels on the edge remain.
    This is used for 3D plotting, where voxels on the inside are not visible
    but still taking up resources by being plotted.

    Parameters
    ----------
    arr : np.ndarray
        3D array to be hollowed out

    Returns
    -------
    np.ndarray
        Hollowed out 3D matrix
    """
    voxels = np.asarray(np.where(arr > 0)).transpose()
    hollowed = np.zeros(arr.shape)

    for x, y, z in voxels:
        area = arr[x - 1:x + (1 + 1), y - 1:y + (1 + 1), z - 1:z + (1 + 1)]

        if np.count_nonzero(area == 0):
            hollowed[x, y, z] = 1

    tmp = arr.copy()
    tmp *= hollowed
    return tmp


def extract_regions(atlas: SpatialImage,
                    region_ids: Union[list, int]) -> SpatialImage:
    """Extract regions from an atlas"""
    if isinstance(region_ids, int):
        region_ids = [region_ids]

    data = atlas.get_data()

    if not np.all(np.isin(region_ids, data)):
        raise ValueError('could not find some (or all) of the given '
                         'region-ids in the atlas')

    data[~np.isin(data, region_ids)] = 0
    data[np.where(data > 0)] = 1

    if len(np.unique(data)) != 2:
        raise ValueError('mask is empty after extracting region from atlas')

    return nib.Nifti1Image(data.astype(int), atlas.affine, atlas.header)


def get_default_target_mask():
    return pkg_resources.resource_filename(__name__,
                                           'templates/MNI152GM.nii.gz')
