#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  Create a project and its associated working directory
"""

from . import __version__
from .utils import get_disk_size, CallCountDecorator, TColor, pyyaml_ordereddict
from .image import binarize_3d, stretch_img, median_filter_img, subtract_img, subsample_img, map_voxels, get_mask_indices
from .fslconf import FSL
from pydoc import locate
from typing import Union
from collections import OrderedDict
from nibabel.processing import resample_from_to, vox2out_vox
import nibabel as nib
import shutil
import numpy as np
import pandas as pd
import pkg_resources
import yaml
import os
import glob
import string
import logging
import time
import sys
import socket


def fail_exit(logfile: str):
    print(TColor.FAIL + 'Project Creation Failed: Resolve all errors before continuing' + TColor.ENDC)
    print(TColor.OKBLUE + 'Log file:' + TColor.ENDC + logfile)
    print(TColor.FAIL + f'{logging.error.count} errors in project' + TColor.ENDC)

    if logging.warning.count > 0:
        print(TColor.WARNING + f'{logging.warning.count} warnings in project' + TColor.ENDC)
    else:
        print(f'{logging.warning.count} warnings in project')

    sys.exit()


def success_exit(stats: dict, work_dir: str, logfile: str):
    print(TColor.OKGREEN + f'New project created in {work_dir}' + TColor.ENDC)
    print(TColor.OKBLUE + 'Log file:' + TColor.ENDC + logfile)
    print(f'{logging.error.count} errors in project')

    if logging.warning.count > 0:
        print(TColor.WARNING + f'{logging.warning.count} warnings in project' + TColor.ENDC)
    else:
        print(f'{logging.warning.count} warnings in project')

    if stats.get('n_bad_participants', 0) > 0:
        print(TColor.WARNING + f'{stats.get("n_bad_participants", 0)} participant(s) removed due to missing data'
              + TColor.ENDC)

    print(f'Manually edit {os.path.join(work_dir, "cluster.json")} to execute the workflow on a cluster '
          f'(e.g., SLURM or qsub)')

    sys.exit()


def get_participant_ids(file: str = None, sep: str = None, index_col: str = 'participant_id') -> list:
    """Load the participants file and see if there are participants in it"""
    if file is None:
        logging.error(f'TypeError: [participants] Input is required')
        return []

    if not os.path.exists(file):
        logging.error(f'FileNotFoundError: [participants] No such file: {file}')
        return []

    participants = list(pd.read_csv(file, sep=sep, engine='python').get(index_col, []))

    if not participants:
        logging.error(f'ValueError: [participants] No participant indices found in {index_col}: {file}')
        return []

    return participants


def validate_paths(d: dict, input_type: str, data: dict, participant_ids: list = None) -> (dict, set):
    """Validate input file formatting and ensure files exist"""

    keys = [key for key, value in d.items() if input_type in value.get('input_type')]

    # Validate filepath format
    for key in keys:
        path = data.get(key, {}).get('file', None) if isinstance(data.get(key, None), dict) else data.get(key, None)
        required = d.get(key, {}).get('required', False)
        template = d.get(key, {}).get('template', False)
        file_type = d.get(key, {}).get('file_type', None)
        expand = d.get(key, {}).get('expand', False)

        if path is None and required:
            logging.error(f'TypeError: [{key}] Input is required')
            continue

        elif path is None and not required:
            logging.info(f'Missing optional input [{key}]')
            continue

        if file_type is not None and not any(path.endswith(ext) for ext in file_type):
            logging.error(f'TypeError: [{key}] {os.path.splitext(path)[-1]} extension given, {file_type} expected')

        if template:
            if 'participant_id' not in [t[1] for t in string.Formatter().parse(path) if t[1] is not None]:
                logging.error(f'TemplateError: [{key}] Missing {{participant_id}} template in {path}')

        else:
            # Non-template files can be checked immediately
            if expand:
                if not (glob.glob(path + '*')):
                    logging.error(f'FileNotFoundError: [{key}] No such file or directory: {path}')

            elif not os.path.exists(path):
                logging.error(f'FileNotFoundError: [{key}] No such file or directory: {path}')

    # Ensure files are present for all participants
    bad_participant_ids = []
    if participant_ids:
        for participant_id in participant_ids:
            for key in keys:
                path = data.get(key, {}).get('file', None) if isinstance(
                    data.get(key, None), dict) else data.get(key, None)

                if d.get(key, {}).get('template', False) and path is not None:
                    path = path.format(participant_id=participant_id)
                    expand = d.get(key, {}).get('expand', False)

                    if expand:
                        if not (glob.glob(path + '*')):
                            logging.warning(f'FileNotFound: [{key}] No such file or directory: {path}')
                            bad_participant_ids.append(participant_id)

                    elif not os.path.exists(path):
                        logging.warning(f'FileNotFound: [{key}] No such file: {path}')
                        bad_participant_ids.append(participant_id)

        bad_participant_ids = set(bad_participant_ids)

    data = {key: data.get(key, None) for key in keys}

    return data, bad_participant_ids


def validate_parameters(d: dict, input_type: str, data: dict) -> dict:
    """Ensure required parameters are given and default values are used where none are entered.

    Returns a data dictionary object with only relevant parameters to the input_type.
    """
    for task in d.keys():
        keys = [key for key, value in d[task].items() if input_type in value.get('input_type')]

        for key in keys:
            if not data.get(task, None):
                data[task] = dict()

            value = data.get(task, {}).get(key, None)
            required = d.get(task, {}).get(key, {}).get('required', False)
            instance_type = d.get(task, {}).get(key, {}).get('instance_type', False)
            default = d.get(task, {}).get(key, {}).get('default', None)
            allowed = d.get(task, {}).get(key, {}).get('allowed', None)

            if required and value is None:
                logging.error(f'TypeError: {task}->{key} requires a value.')
                continue

            elif not required and value is None and default is not None:
                logging.info(f'DefaultValue: Setting {task}->{key} to {default}')
                data[task][key] = default
                continue

            if allowed is not None:
                if isinstance(value, list) and not set(value).issubset(allowed):
                    logging.error(f'InputError: [{task}->{key}] Must be {", ".join(allowed)}, not {value}')
                    continue

                elif isinstance(value, str) and value not in allowed:
                    logging.error(f'InputError: [{task}->{key}] Must be {", ".join(allowed)}, not {value}')
                    continue

            if type(value) is not locate(instance_type) and value is not None:
                logging.error(f'TypeError: [{task}->{key}] Must be {type(value).__name__}, not {instance_type}')
                continue

        data[task] = {key: data.get(task, {}).get(key, None) for key in keys}

    if input_type == 'rsfmri':
        # Ensure low_pass > high_pass
        low_pass = data.get('connectivity', {}).get('low_pass', None)
        high_pass = data.get('connectivity', {}).get('high_pass', None)
        tr = data.get('connectivity', {}).get('tr', None)

        if low_pass is not None and high_pass is not None:
            if high_pass >= low_pass:
                logging.error(f'ValueError: High-pass ({high_pass}) is expected to be smaller than low-pass '
                              f'({low_pass})')

            if tr is None:
                logging.error('ValueError: connectivity->tr requires a value')

    return data


def load_img(name: str, mask: str) -> Union[nib.spatialimages.SpatialImage, bool]:
    """Check if the input mask can be read using nibabel"""
    try:
        return nib.load(mask)
    except Exception as e:
        logging.error(f'ValueError: [{name}_mask] Unable to read contents of file: {mask}')
        return False


def process_seed_indices(config: dict) -> Union[dict, bool]:
    indices_file = config.get('input_data', {}).get('seed_indices', None)
    seed = config.get('input_data', {}).get('seed_mask', None)

    try:
        indices = np.load(indices_file)
    except Exception as e:
        logging.error(f'ValueError: [seed_indices] Unable to read contents of file: {indices_file}')
        return False

    seed_img = load_img('seed', seed)
    n_voxels = np.count_nonzero(seed_img.get_data())

    if indices.shape != (n_voxels, 3):
        logging.error(f'ValueError: [seed_indices] Expected shape ({n_voxels}, 3), not {indices.shape}')
        return False

    files = {'seed_mask': seed_img, 'seed_indices': indices}
    return files


def process_masks(config: dict) -> Union[dict, bool]:
    def _base_processing(file, img, resample_to_mni: bool = False, bin_threshold: float = 0.0):
        mni_mapped_voxels = map_voxels(
            voxel_size=[2, 2, 2],
            origin=[90, -126, -72],
            shape=(91, 109, 91)
        )

        data = img.get_data()

        # Binarize
        if not np.array_equal(data, data.astype(bool)):
            img = binarize_3d(img, threshold=bin_threshold)
            logging.warning(f'Binarizing {file}: setting all values >{bin_threshold} to 1 and others to 0')

        # Resample
        if resample_to_mni:
            shape = img.shape
            affine = img.affine
            mapped_voxels = mni_mapped_voxels

            if not np.all([np.all(np.equal(a, b)) for a, b in zip(mapped_voxels, (shape, affine))]):
                img = resample_from_to(img, mapped_voxels, order=0, mode='nearest')
                logging.warning(f'Resampling {file} to MNI group template '
                                f'(nibabel.processing.resample_from_to), using order=0, mode=\'nearest\'')

        return img

    # Input data
    seed = config.get('input_data', {}).get('seed_mask', None)
    target = config.get('input_data', {}).get('target_mask', None)

    # Parameters
    input_data_type = config.get('input_data_type')
    options = config.get('parameters', {}).get('masking', {})
    resample_to_mni = options.get('resample_to_mni', False)
    bin_threshold = options.get('threshold', 0.0)
    median_filter = options.get('median_filter', False)
    median_filter_dist = options.get('median_filter_dist', 1)
    upsample = options.get('upsample_seed_to', None)
    del_seed_from_target = options.get('del_seed_from_target', False)
    del_seed_dist = options.get('del_seed_dist', 0)
    subsample = options.get('subsample', False)
    downsample = options.get('downsample_target_to', None)

    # Load images
    seed_img = load_img('seed', seed)

    if target is None:
        # Load default MNI152GM template
        target = 'Default MNI152 Gray Matter template'
        target_img = nib.load(pkg_resources.resource_filename(__name__, 'templates/MNI152GM.nii'))
        logging.warning(f'DefaultValue: [target_mask] Using default MNI152 Gray Matter template')

    else:
        target_img = load_img('target', target)

    seed_img = _base_processing(seed, seed_img, resample_to_mni=resample_to_mni, bin_threshold=bin_threshold)
    target_img = _base_processing(target, target_img, resample_to_mni=resample_to_mni, bin_threshold=bin_threshold)

    if logging.error.count > 0:
        return False

    # Process seed
    if median_filter:
        seed_img = median_filter_img(seed_img, dist=median_filter_dist)
        logging.info(f'Applying median filter on seed ({seed}) (dist={median_filter_dist})')

    if input_data_type == 'dmri':
        if upsample is not None:
            if len(upsample) == 1:
                upsample = [upsample] * 3

            if len(upsample) != 3:
                logging.error(f'ValueError: [masking->upsample_seed_to] Requires 1 or 3 values, not {len(upsample)}')

            else:
                mapped_voxels = (seed_img.shape, seed_img.affine)
                highres_seed_img = stretch_img(source_img=seed_img, target=vox2out_vox(mapped_voxels, upsample))
                logging.info(f'Stretched seed ({seed}) to fit a '
                             f'{"x".join(map(str, highres_seed_img.shape))} template with '
                             f'{"x".join(map(str, upsample))} voxel size')

    # Process target
    if input_data_type == 'rsfmri':
        # Remove seed voxels from target
        if del_seed_from_target:
            target_img = subtract_img(
                source_img=target_img,
                target_img=seed_img,
                edge_dist=del_seed_dist
            )
            logging.info(f'Removing seed ({seed}) from target ({target}) (edge_dist={del_seed_dist}')

        # Reduce the number of voxels in target
        if subsample:
            target_img = subsample_img(img=target_img, f=2)
            logging.info(f'Subsampling target image ({target})')

    elif input_data_type == 'dmri':
        if downsample is not None:
            if len(downsample) == 1:
                downsample = [downsample] * 3

            if len(downsample) != 3:
                logging.error(f'ValueError: [masking->downsample_target_to] Requires 1 or 3 values, not '
                              f'{len(downsample)}')

            else:
                mapped_voxels = vox2out_vox((target_img.shape, target_img.affine), downsample)
                target_img = resample_from_to(target_img, mapped_voxels, order=0, mode='nearest')
                logging.warning(f'Resampling {target} to {"x".join(map(str, downsample))} voxel size '
                                f'(nibabel.processing.resample_from_to), using order=0, mode=\'nearest\'')

    if logging.error.count > 0:
        return False

    masks = {'seed_mask': seed_img, 'target_mask': target_img}

    if input_data_type == 'dmri' and upsample is not None:
        masks['highres_seed_mask'] = highres_seed_img

    return masks


def create_workflow(config: dict, work_dir: str) -> None:
    def get_value(data: dict, *args: str) -> str:
        return data.get(args[0], None) if len(args) == 1 else get_value(data.get(args[0], {}), *args[1:])

    input_data_type = config.get('input_data_type')
    templates = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'templates')
    snakefiles = ['header.Snakefile', 'body.Snakefile']

    if input_data_type in ('rsfmri', 'dmri'):
        snakefiles.insert(1, f'{input_data_type}.Snakefile')
        config['input_data']['connectivity_matrix'] = 'connectivity/connectivity_{participant_id}.npy'
        config['input_data']['seed_indices'] = ''

    if input_data_type == 'rsfmri':
        config['input_data']['touchfile'] = 'log/.touchfile'

    with open(os.path.join(work_dir, 'Snakefile'), 'w') as of:
        for snakefile in snakefiles:
            with open(os.path.join(templates, snakefile), 'r') as f:
                for line in f:
                    if line.find('<cbptools[') != -1:
                        template = line[line.find("<cbptools['") + 11:line.find("']>")]
                        keys = template.split(':')
                        value = get_value(config, *keys)

                        if isinstance(value, dict):
                            value = value if not value.get('file', None) else value.get('file')

                        if not (keys[0] == 'input_data' and not value):
                            value = repr(value) if isinstance(value, str) else str(value)
                            line = line.replace(f"<cbptools['{template}']>", f'{keys[-1]} = {value}')
                            of.write(line)

                    elif line.find('<!cbptools[') != -1:
                        template = line[line.find("<!cbptools['") + 12:line.find("']>")]
                        keys = template.split(':')
                        value = get_value(config, *keys)
                        line = line.replace(f"<!cbptools['{template}']>", str(value))
                        of.write(line)

                    else:
                        of.write(line)

    shutil.copy(os.path.join(templates, 'cluster.json'), work_dir)


def create_project(work_dir, config, participants: pd.DataFrame, files: dict = None) -> dict:
    # # Save config
    # with open(os.path.join(work_dir, 'project.yaml'), 'w') as stream:
    #     yaml.dump(config, stream, default_flow_style=False)

    # Save files
    for key, value in files.items():
        if isinstance(value, nib.spatialimages.SpatialImage):
            nib.save(value, os.path.join(work_dir, key + '.nii'))
            logging.info(f'Created file {os.path.join(work_dir, key + ".nii")}')

        elif isinstance(value, np.ndarray):
            np.save(os.path.join(work_dir, key + '.npy'), value)
            logging.info(f'Created file {os.path.join(work_dir, key + ".npy")}')

    # Save participant info
    n_bad_participants = np.count_nonzero(participants.invalid)
    n_participants = participants.participant_id.count()
    participants = pd.DataFrame(participants[~participants['invalid']]['participant_id'])
    participants.to_csv(os.path.join(work_dir, 'participants.tsv'), sep='\t', index=False)

    # Create workflow (snakefile)
    create_workflow(config=config, work_dir=work_dir)

    # Info
    input_data_type = config.get('input_data_type')

    if input_data_type in ('rsmfri', 'dmri'):
        seed_voxels = (np.asarray(masks['seed_mask'].get_data()) == 1).sum()
        target_voxels = (np.asarray(masks['target_mask'].get_data()) == 1).sum()
        n_participants_included = n_participants-n_bad_participants
        connectivity_size = seed_voxels*target_voxels*n_participants_included
        cluster_labels_size = (seed_voxels*n_participants_included) + (seed_voxels*2) + n_participants_included + 1

        logging.info(f'Included participants: {n_participants_included}')
        logging.info(f'Approximate size for connectivity results: {get_disk_size(connectivity_size, 8)}')
        logging.info(f'Approximate size for clustering results: {get_disk_size(cluster_labels_size, 8)}')
        logging.info(f'Size of summary results cannot be approximated')

    stats = {
        'n_bad_participants': n_bad_participants,
        'n_participants_total': n_participants
    }

    return stats


def validate_config(configfile: str, work_dir: str, logfile: str):
    """Validate configuration file. Returns false if validation cannot continue"""

    # make pyyaml work with OrderedDict
    yaml.add_representer(OrderedDict, pyyaml_ordereddict)

    # Set up logging
    if os.path.exists(os.path.join(work_dir, 'log', 'project.log')):
        os.remove(os.path.join(work_dir, 'log', 'project.log'))

    logging.basicConfig(
        filename=logfile,
        format='%(asctime)s\t%(levelname)s\t%(message)s',
        datefmt='%I:%M:%S%p',
        level=logging.INFO
    )
    logging.error = CallCountDecorator(logging.error)
    logging.warning = CallCountDecorator(logging.warning)
    logging.info(f'CBP tools version {__version__}')
    logging.info(f'Setup initiated on {time.strftime("%b %d %Y %H:%M:%S")} in environment {sys.prefix}')
    logging.info(f'Username of creator is "{os.getlogin()}" with hostname "{socket.gethostname()}"')

    # Load configfile
    with open(configfile, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(f'Could not load configfile: {exc}')
            return False, False

    # Load predefined settings
    with open(pkg_resources.resource_filename(__name__, 'defaults.yaml'), 'r') as stream:
        defaults = yaml.safe_load(stream)

    # Validate input data type
    input_data_type = config.get('input_data_type', None)

    if input_data_type not in defaults.get('input_data_types'):
        logging.error(f'TypeError: [input_data_type] Must be {", ".join(defaults.get("input_data_types"))}, not'
                      f'{input_data_type}')
    else:
        input_data_type = config['input_data_type'] = input_data_type.lower()

    # Validate input data
    input_data = config.get('input_data', {})
    participant_ids = []
    participant_ids_bad = []

    if not input_data:
        logging.error(f'ValueError: [input_data] No {input_data_type} input data found')

    else:
        participant_ids = get_participant_ids(
            file=input_data.get('participants').get('file', None),
            sep=input_data.get('participants', {}).get('sep', None),
            index_col=input_data.get('participants', {}).get('index_col', 'participant_id')
        )

        input_data, participant_ids_bad = validate_paths(
            d=defaults.get('input'),
            input_type=input_data_type,
            data=input_data,
            participant_ids=participant_ids
        )

        if participant_ids and len(set(participant_ids) - participant_ids_bad) <= 1:
            logging.error('ValueError: [participants] After removing participants with missing data, none are left')

    # Validate parameters
    parameters = config.get('parameters', {})
    if not parameters:
        logging.error(f'ValueError: [parameters] No {input_data_type} parameters found')

    else:
        parameters = validate_parameters(
            d=defaults.get('parameters'),
            input_type=input_data_type,
            data=parameters
        )

    # Check if FSL is accessible
    if input_data_type == 'dmri':
        fsl = FSL()
        if fsl.has_probtrackx2():
            logging.info(f'FSLInfo: Executable path is \'{fsl.fsl}\'')
            logging.info(f'FSLInfo: probtrackx2 executable path is \'{fsl.probtrackx2}\'')
            logging.info(f'FSLInfo: Directory is {fsl.fsl_dir}')
            logging.info(f'FSLInfo: Output type is {fsl.fsl_outputtype}')

        else:
            logging.warning('ModuleNotFoundError: No module named \'probtrackx2\' (FSL)')

    if logging.error.count > 0:
        return False

    participants = pd.DataFrame(set(participant_ids), columns=['participant_id'])
    participants.sort_values(by='participant_id', inplace=True)
    participants['invalid'] = np.where(participants['participant_id'].isin(participant_ids_bad), True, False)

    config = OrderedDict({
        'input_data_type': input_data_type,
        'input_data': input_data,
        'parameters': parameters
    })

    # Process Masks
    if input_data_type in ('rsfmri', 'dmri'):
        files = process_masks(config=config)

    else:
        files = process_seed_indices(config=config)

    if not files or logging.error.count > 0:
        return False

    info = create_project(work_dir=work_dir, config=config, files=files, participants=participants)
    logging.info(f'Project setup completed on {time.strftime("%b %d %Y %H:%M:%S")}')
    logging.shutdown()  # no more log entries are made

    return info
