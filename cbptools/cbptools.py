#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  Create a project and its associated working directory
"""

from . import __version__
from .utils import readable_bytesize, CallCountDecorator, TColor, pyyaml_ordereddict, bytes_to
from .image import binarize_3d, stretch_img, median_filter_img, subtract_img, subsample_img, map_voxels
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
    n_errors = logging.error.count
    n_warnings = logging.warning.count
    print(f'{TColor.FAIL}Project Creation Failed: Resolve all errors before continuing{TColor.ENDC}')
    print(f'{TColor.OKBLUE}Log file:{TColor.ENDC} {logfile}')
    print(f'{TColor.FAIL}{n_errors} errors in project{TColor.ENDC}')

    if n_warnings > 0:
        print(f'{TColor.WARNING}{n_warnings} warnings in project{TColor.ENDC}')
    else:
        print(f'{n_warnings} warnings in project')

    sys.exit()


def success_exit(stats: dict, work_dir: str, logfile: str):
    n_errors = logging.error.count
    n_warnings = logging.warning.count

    print(f'{TColor.OKGREEN}New project created in {work_dir}{TColor.ENDC}')
    print(f'{TColor.OKBLUE}Log file:{TColor.ENDC} {logfile}')
    print(f'{n_errors} errors in project')

    if n_warnings > 0:
        print(f'{TColor.WARNING}{n_warnings} warnings in project{TColor.ENDC}')
    else:
        print(f'{n_warnings} warnings in project')

    if stats.get('n_bad_participants', 0) > 0:
        print(f'{TColor.WARNING}{stats.get("n_bad_participants")} participant(s) '
              f'removed due to missing data{TColor.ENDC}')

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


def estimate_memory_usage(config: dict, files: dict, participants: list) -> dict:
    input_data_type = config.get('input_data_type')
    mem_mb = dict()

    # Connectivity task
    if input_data_type == 'rsfmri':
        buffer = 250  # in MB
        time_series = config.get('input_data', {}).get('time_series')
        sizes = [os.path.getsize(time_series.format(participant_id=participant)) for participant in participants]
        mem_mb['connectivity'] = int(np.ceil(bytes_to(np.ceil(max(sizes)*2.5), 'mb'))) + buffer

    elif input_data_type == 'dmri':
        mem_mb['connectivity'] = 10000  # TODO: Figure out probtrackx2 memory usage

    if input_data_type in ('rsfmri', 'dmri'):
        # Clustering task
        buffer = 250
        seed_voxels = (np.asarray(files['seed_mask'].get_data()) == 1).sum()
        target_voxels = (np.asarray(files['target_mask'].get_data()) == 1).sum()
        mem_mb['clustering'] = int(np.ceil(bytes_to(seed_voxels * target_voxels * len(participants), 'mb'))) + buffer

    elif input_data_type == 'connectivity':
        buffer = 250
        connectivity = config.get('input_data', {}).get('connectivity_matrix')
        sizes = [os.path.getsize(connectivity.format(participant_id=participant)) for participant in participants]
        mem_mb['clustering'] = int(np.ceil(bytes_to(np.ceil(max(sizes)), 'mb'))) + buffer

    return mem_mb


def validate_paths(d: dict, input_type: str, data: dict, participant_ids: list = None) -> dict:
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
    participant_ids_bad = []
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
                            participant_ids_bad.append(participant_id)

                    elif not os.path.exists(path):
                        logging.warning(f'FileNotFound: [{key}] No such file: {path}')
                        participant_ids_bad.append(participant_id)

        participant_ids_bad = set(participant_ids_bad)

    data = {key: data.get(key, None) for key in keys}
    data['participant_ids_bad'] = participant_ids_bad
    return data


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

    # Ensure low_pass > high_pass for rsfmri data
    if input_type == 'rsfmri':
        low_pass = data.get('connectivity', {}).get('low_pass', None)
        high_pass = data.get('connectivity', {}).get('high_pass', None)
        tr = data.get('connectivity', {}).get('tr', None)

        if low_pass is not None and high_pass is not None:
            if high_pass >= low_pass:
                logging.error(f'ValueError: High-pass ({high_pass}) is expected to be smaller than low-pass '
                              f'({low_pass})')

            if tr is None:
                logging.error('ValueError: connectivity->tr requires a value')

    # Reset values so that they can be used for FSL input directly
    elif input_type == 'dmri':
        if data.get('connectivity', {}).get('correct_path_distribution'):
            data['connectivity']['correct_path_distribution'] = '--pd'

        else:
            data['connectivity']['correct_path_distribution'] = ''

        if data.get('connectivity', {}).get('loop_check'):
            data['connectivity']['loop_check'] = '-l'

        else:
            data['connectivity']['loop_check'] = ''

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

    except Exception as exc:
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
    def base_proc(file, img, resample_to_mni: bool = False, bin_threshold: float = 0.0):
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

    # Load default MNI152GM template if target is not given
    if target is None:
        target = 'Default MNI152 Gray Matter template'
        target_img = nib.load(pkg_resources.resource_filename(__name__, 'templates/MNI152GM.nii'))
        logging.warning(f'DefaultValue: [target_mask] Using default MNI152 Gray Matter template')

    else:
        target_img = load_img('target', target)

    seed_img = base_proc(seed, seed_img, resample_to_mni=resample_to_mni, bin_threshold=bin_threshold)
    target_img = base_proc(target, target_img, resample_to_mni=resample_to_mni, bin_threshold=bin_threshold)

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


def create_workflow(config: dict, mem_mb: dict, work_dir: str) -> None:
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

    config['mem_mb'] = mem_mb

    with open(os.path.join(work_dir, 'Snakefile'), 'w') as of:
        for snakefile in snakefiles:
            with open(os.path.join(templates, snakefile), 'r') as f:
                for line in f:
                    tags = ({'start': "<cbptools[\'", 'end': "\']>"}, {'start': "<!cbptools[\'", 'end': "\']>"})

                    if line.find(tags[0].get('start')) != -1:
                        s, e = tags[0].get('start'), tags[0].get('end')
                        content = line[line.find(s) + len(s):line.find(e)]
                        content, inplace = (content[1:], True) if content.startswith('!') else (content, False)
                        keys = content.split(':')
                        value = get_value(config, *keys)

                        if isinstance(value, dict):
                            value = value if not value.get('file', None) else value.get('file')

                        if keys[0] == 'input_data' and not value:
                            continue

                        if inplace:
                            line = line.replace(f'{s}!{content}{e}', str(value))
                        else:
                            value = repr(value) if isinstance(value, str) else str(value)
                            line = line.replace(f'{s}{content}{e}', f'{keys[-1]} = {value}')

                    of.write(line)

    shutil.copy(os.path.join(templates, 'cluster.json'), work_dir)


def create_project(work_dir: str, config: dict, files: dict, mem_mb: dict, participants: pd.DataFrame) -> dict:
    """Generate all the files needed by the CBP project"""
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
    create_workflow(config=config, mem_mb=mem_mb, work_dir=work_dir)

    # Info
    input_data_type = config.get('input_data_type')
    n_participants_included = n_participants - n_bad_participants
    logging.info(f'Included participants: {n_participants_included}')
    seed_voxels = (np.asarray(files['seed_mask'].get_data()) == 1).sum()

    if input_data_type in ('rsfmri', 'dmri'):
        target_voxels = (np.asarray(files['target_mask'].get_data()) == 1).sum()
        connectivity_size = seed_voxels*target_voxels*n_participants_included
        logging.info(f'Approximate size of all connectivity matrices: {readable_bytesize(connectivity_size, 8)}')

    cluster_labels_size = (seed_voxels * n_participants_included) + (seed_voxels * 2) + n_participants_included + 1
    logging.info(f'Approximate size of all cluster label files: {readable_bytesize(cluster_labels_size, 8)}')

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

        input_data = validate_paths(
            d=defaults.get('input'),
            input_type=input_data_type,
            data=input_data,
            participant_ids=participant_ids
        )

        participant_ids_bad = input_data['participant_ids_bad']

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

    # Get estimated memory usage of tasks
    mem_mb = estimate_memory_usage(
        config=config,
        files=files,
        participants=list(set(participant_ids) - set(participant_ids_bad))
    )

    info = create_project(work_dir=work_dir, config=config, files=files, mem_mb=mem_mb, participants=participants)
    logging.info(f'Project setup completed on {time.strftime("%b %d %Y %H:%M:%S")}')
    logging.shutdown()  # no more log entries are made

    return info


def copy_example(params, **kwargs):
    """Copy an example configuration file to the current working directory"""
    input_data_type = params.input_data_type
    filename = f'config_{input_data_type}.yaml'
    file = pkg_resources.resource_filename(__name__, f'templates/{filename}')
    dest = os.path.join(os.getcwd(), filename)

    if os.path.exists(dest):
        path, ext = os.path.splitext(dest)
        dest = f'{path} ({{i}}){ext}'
        i = 0
        while os.path.exists(dest.format(i=i)):
            i += 1

        dest = dest.format(i=i)

    shutil.copy(file, dest)
    print(f'Copied {input_data_type} example configuration file to {dest}')
