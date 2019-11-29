from ..image import (subsample_img, binarize_3d, median_filter_img,
                     subtract_img, get_mask_indices, stretch_img,
                     extract_regions)
from ..utils import get_logger
from nibabel.processing import resample_from_to, vox2out_vox
import pkg_resources
import nibabel as nib
import numpy as np


def process_masks_rsfmri(input: dict, output: dict, params: dict,
                         log: list) -> None:
    seed_mask = input.get('seed_mask')
    target_mask = input.get('target_mask', None)
    out_seed = output.get('seed_img')
    out_target = output.get('target_img')
    out_coordinates = output.get('seed_coordinates')
    log_file = log[0]
    region_id = params.get('region_id', None)
    binarize_seed = params.get('bin_seed', None)
    binarize_target = params.get('bin_target', None)
    median_filter = params.get('median_filter', False)
    remove_seed = params.get('remove_seed', False)
    subsampling = params.get('subsample', False)

    # Set up logging
    logger = get_logger('process_masks_rsfmri', log_file)

    if target_mask is None:
        logger.info('loading template 2mm^3 MNI152 gray matter mask as target')
        target_mask = pkg_resources.resource_filename(
            __name__, 'templates/MNI152GM.nii.gz')

    seed_img = nib.load(seed_mask)
    target_img = nib.load(target_mask)

    # Seed is atlas, extract regions
    if region_id and seed_img is not None:
        logger.info('extracting seed mask from atlas with region_id: %s'
                    % str(region_id).strip('[]'))
        seed_img = extract_regions(seed_img, region_id)

    # Perform binarization
    if binarize_seed:
        seed_img = binarize_3d(seed_img, threshold=binarize_seed)

    if binarize_target:
        target_img = binarize_3d(target_img, threshold=binarize_target)

    # Median filter the seed
    if median_filter:
        logger.info('applying median filter on seed_mask (distance=%s)'
                    % median_filter)
        seed_img = median_filter_img(seed_img, dist=median_filter)

    # Remove seed from target
    if remove_seed:
        logger.info('removing seed_mask from target_mask (distance=%s)'
                    % remove_seed)
        target_img = subtract_img(target_img, seed_img, remove_seed)

    # Subsample target
    if subsampling:
        logger.info('applying subsampling on target_mask')
        target_img = subsample_img(target_img, f=2)

    # Obtain seed coordinates
    seed_coordinates = get_mask_indices(seed_img, order='C')

    # save output
    nib.save(seed_img, out_seed)
    nib.save(target_img, out_target)
    np.save(out_coordinates, seed_coordinates)


def process_masks_dmri(input: dict, output: dict, params: dict,
                       log: list) -> None:
    seed_mask = input.get('seed_mask')
    target_mask = input.get('target_mask', None)
    out_seed = output.get('seed_img')
    out_target = output.get('target_img')
    out_coordinates = output.get('seed_coordinates')
    out_highres_seed = output.get('highres_seed_img', None)
    log_file = log[0]
    region_id = params.get('region_id', None)
    binarize_seed = params.get('bin_seed', None)
    binarize_target = params.get('bin_target', None)
    median_filter = params.get('median_filter', False)
    remove_seed = params.get('remove_seed', False)
    upsample_to = params.get('upsample_to', False)
    downsample_to = params.get('downsample_to', False)

    # Set up logging
    logger = get_logger('process_masks_dmri', log_file)

    if target_mask is None:
        logger.info('loading template 2mm^3 MNI152 gray matter mask as target')
        target_mask = pkg_resources.resource_filename(
            __name__, 'templates/MNI152GM.nii.gz')

    seed_img = nib.load(seed_mask)
    target_img = nib.load(target_mask)

    # Seed is atlas, extract regions
    if region_id and seed_img is not None:
        logger.info('extracting seed mask from atlas with region_id: %s'
                    % str(region_id).strip('[]'))
        seed_img = extract_regions(seed_img, region_id)

    # Perform binarization
    if binarize_seed:
        seed_img = binarize_3d(seed_img, threshold=binarize_seed)

    if binarize_target:
        target_img = binarize_3d(target_img, threshold=binarize_target)

    # Median filter the seed
    if median_filter:
        logger.info('applying median filter on seed_mask (distance=%s)'
                    % median_filter)
        seed_img = median_filter_img(seed_img, dist=median_filter)

    # Remove seed from target
    if remove_seed:
        logger.info('removing seed_mask from target_mask (distance=%s)'
                    % remove_seed)
        target_img = subtract_img(target_img, seed_img, remove_seed)

    # Upsample seed
    if upsample_to:
        if len(upsample_to) == 1:
            upsample_to *= 3

        logger.info('stretching seed_mask (highres_seed_mask) to %s'
                    % 'x'.join(map(str, upsample_to)))

        mapping = list(
            vox2out_vox((seed_img.shape, seed_img.affine), upsample_to))
        a = np.sign(seed_img.affine)
        b = np.sign(mapping[1])
        mapping[1] *= (a * b)
        highres_seed = stretch_img(seed_img, mapping)
        nib.save(highres_seed, out_highres_seed)

    # Downsample target
    if downsample_to:
        if len(downsample_to) == 1:
            downsample_to *= 3

        logger.info('downsampling target_mask to %s'
                    % 'x'.join(map(str, upsample_to)))

        mapping = list(vox2out_vox((
            target_img.shape, target_img.affine), downsample_to))
        a = np.sign(target_img.affine)
        b = np.sign(mapping[1])
        mapping[1] *= (a * b)
        target_img = resample_from_to(target_img, mapping, order=0,
                                      mode='nearest')

    # Obtain seed coordinates
    seed_coordinates = get_mask_indices(seed_img, order='C')

    # save output
    nib.save(seed_img, out_seed)
    nib.save(target_img, out_target)
    np.save(out_coordinates, seed_coordinates)
