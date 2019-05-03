#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for processing neuroimaging data.
Expected input to these functions are np.ndarrays rather than nibabel spatial images. This cuts down in computation
cost for loading data and formatting the data back into spatial image objects."""

from .exceptions import ShapeError, DimensionError
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

    if not _check_equal(affines) or not _check_equal(shapes):
        return False
    else:
        return True


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


def img_is_mask(img: np.ndarray, allow_empty: bool=True) -> bool:
    """Check if input array meets the criteria for being a mask (3D, binary, not empty)"""
    if not img_is_3d(img):
        return False
    elif not ((img == 0) | (img == 1)).all() and img.dtype is not bool:
        return False
    elif np.sum(img) == 0 and not allow_empty:
        return False
    return True


def binarize_3d(img: spatialimage, threshold: float = 0.0) -> spatialimage:
    """ binarize 3D spatial image """
    if img_is_3d(img):
        return nib.Nifti1Image(np.where(img.get_data() > threshold, 1, 0), img.affine, img.header)
    else:
        raise ShapeError(3, len(img.shape))


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
    if img_is_3d(img):
        data = img.get_data()
        mask = np.zeros(img.shape)
        mask[::f, ::f, ::f] = 1
        data *= mask
        return nib.Nifti1Image(data, img.affine, img.header)
    else:
        raise ShapeError(3, len(img.shape))


def median_filter_img(img: spatialimage, dist: int = 1) -> spatialimage:
    """Median filtering of non-zero elements in an image.

    Median filter all selected voxels of a binary 3D input image and a 2-iteration dilation border around it. Voxel
    distance dist is used to determine the size of the area for computing the median, where a distance of 1 results in
    a 3*3*3 shape. The number of repetitions nrep determines how often the median filter will be repeated.
    """

    if img_is_3d(img):
        data = img.get_data()
        dilated = binary_dilation(data, iterations=2).astype(int)
        voxels = np.asarray(np.where(dilated == 1)).transpose()

        filtered = np.zeros(img.shape)
        for x, y, z in voxels:
            area = data[x - dist:x + (dist + 1), y - dist:y + (dist + 1), z - dist:z + (dist + 1)]

            if np.median(area) > float_info.min:
                filtered[x, y, z] = 1

        return nib.Nifti1Image(np.float32(filtered), img.affine, img.header)
    else:
        raise ShapeError(3, len(img.shape))


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


def apply_mask(time_series: spatialimage, mask_img: spatialimage, as_array: bool=True):
    """ Apply a 3D mask to a 4D image

    Parameters
    ----------
    time_series : spatialimage
        4D nifti image, with time-series on the 4th dimension
    mask_img : spatialimage
        3D boolean mask image
    as_array : bool
        Returns a data array if True, otherwise a spatialimage
    """

    if not imgs_equal_3d([time_series, mask_img]):
        raise ValueError('Input time-series and mask do not have equal shape and/or affine')

    if not img_is_4d(time_series):
        raise ShapeError(4, len(time_series.shape))

    time_series_data = time_series.get_data()
    mask_img_data = mask_img.get_data().astype(bool)

    if as_array:
        return time_series_data[mask_img_data].T
    else:
        return nib.Nifti1Image(time_series_data[mask_img_data].T, mask_img.affine, mask_img.header)


def apply_masks(img: spatialimage, masks: List[spatialimage]) -> [np.ndarray, List[List[int]]]:
    """ Apply multiple masks to one spatial image time series object

    Masks are 'stacked' resulting in one masked output matrix and multiple mask indices that allow retrieval of
    individual mask data at later points in time. This method stores only one matrix in memory and allows preprocessing
    operations to be applied to voxels contained in both masks at once.

    Notes
    -----
    Seeing as how the output type is np.ndarray, the affine and header of the input image is not preserved.

    Parameters
    ----------
    img : Union[str, nibabel.spatialimages.SpatialImage]
        4D image with time series that is to be masked.
    masks : List[Union[str, spatialimage]]
        A list containing the file paths pointing to each mask image

    Returns
    -------
    [np.ndarray, List[List[int]]]
        The merged mask and a list of indices for each individual mask.

    """

    imgs = [img] + masks

    if not imgs_equal_3d(imgs=imgs):
        raise ValueError('Input image and masks do not have equal shape and/or affine')

    if len(imgs[0].shape) != 4:
        raise DimensionError(len(imgs[0].shape), 4)

    indices = [img.get_data().flatten(order='F').nonzero()[0].tolist() for img in imgs[1:]]
    mask_stack = np.unique(np.concatenate(indices))
    indices = [np.where(np.in1d(mask_stack, index))[0].tolist() for index in indices]

    data = imgs[0].get_data().reshape(-1, imgs[0].shape[-1], order='F').T.astype(np.float32)
    data = np.take(data, mask_stack, axis=1).squeeze()

    return data, indices


def unmask(labels: Union[List[int], np.ndarray], mask: spatialimage, order: str='C') -> spatialimage:
    """Takes a 1D array as labels and a 3D mask image and places the labels into the mask based on the given sorting
    order.

    Notes
    -----
    This function should only be applied to data sorted with order 'F', as order sorting orders are slow and should use
    numpy's default indexing.

    Parameters
    ----------
    labels : Union[List[int], np.ndarray]
        List or numpy array containing voxel values. The voxel order should be as if it were extracted from the mask by
        using order 'F'
    mask : Union[str, nibabel.spatialimages.SpatialImage]
        3D mask image where the label data will be put into
    order : str
        ‘C’ means to use row-major (C-style) order. ‘F’ means to use column-major (Fortran-style) order.

    Returns
    -------
    nibabel.spatialimages.SpatialImage
        3D nifti image where the label values have been placed into the mask voxels

    """

    labels = np.asarray(labels).astype(int)
    mask_data = mask.get_data()

    if not img_is_mask(mask_data, allow_empty=False):
        raise ValueError('Input mask is not a spatial image mask.')

    img = mask_data.flatten(order=order).astype(int)
    np.place(img, [img == 1], labels)
    img = np.reshape(img, mask.shape, order=order)
    return nib.Nifti1Image(np.float32(img), mask.affine, mask.header)
