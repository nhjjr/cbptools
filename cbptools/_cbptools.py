#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a project and its associated working directory"""

from . import __version__
from .utils import readable_bytesize, CallCountDecorator, TColor, \
    pyyaml_ordereddict, bytes_to
from .image import binarize_3d, stretch_img, median_filter_img, subtract_img, \
    subsample_img, map_voxels, imgs_equal_3d, get_mask_indices
from nibabel.processing import resample_from_to, vox2out_vox
from collections import OrderedDict
from pydoc import locate
from typing import Union
from fnmatch import fnmatch
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

spatialimage = nib.spatialimages.SpatialImage


def fail_exit(logfile: str):
    n_errors = logging.error.count
    n_warnings = logging.warning.count
    lines = '\n'.join((
        '  {fail}Project Creation Failed: Resolve all errors before '
        'continuing{endc}',
        '  {ok}Log file:{endc} {logfile}',
        '  {fail}{n_errors} errors in project{endc}',
        '  {warning}{n_warnings} warnings in project{endc}'
    )).format(
        fail=TColor.FAIL,
        endc=TColor.ENDC,
        ok=TColor.OKBLUE,
        warning=TColor.WARNING if n_warnings > 0 else TColor.ENDC,
        logfile=logfile,
        n_errors=n_errors,
        n_warnings=n_warnings
    )
    print('\n' + lines + '\n')
    sys.exit()


def success_exit(stats: dict, work_dir: str, logfile: str):
    n_errors = logging.error.count
    n_warnings = logging.warning.count
    n_bad_participants = stats.get('n_bad_participants', 0)
    cluster_json = os.path.join(work_dir, "cluster.json")

    lines = '\n'.join((
        '  {okgreen}New project created in {work_dir}{endc}',
        '  {okblue}Log file:{endc} {logfile}',
        '  {n_errors} errors in project',
        '  {warning}{n_warnings} warnings in project{endc}',
        '  {pwarning}{n_bad_participants} participant(s) removed due to '
        'missing data{endc}',
        '',
        '  Manually edit {cluster_json} to execute the workflow on a cluster '
        'environment (e.g., SLURM or qsub)'
    )).format(
        okgreen=TColor.OKGREEN,
        work_dir=work_dir,
        endc=TColor.ENDC,
        okblue=TColor.OKBLUE,
        logfile=logfile,
        n_errors=n_errors,
        warning=TColor.WARNING if n_warnings > 0 else TColor.ENDC,
        n_warnings=n_warnings,
        pwarning=TColor.WARNING if n_bad_participants > 0 else TColor.ENDC,
        n_bad_participants=n_bad_participants,
        cluster_json=cluster_json
    )
    print('\n' + lines + '\n')
    sys.exit()


def get_filepath(value: Union[str, dict]):
    if isinstance(value, dict):
        value = value.get('file')

    return value


def parse_line(line: str, config: dict) -> Union[str, bool]:
    """Parse <cbptools['key1:key2:key3']> string"""

    def get_value(data: dict, *args: str) -> Union[str, None]:
        if not isinstance(data, dict):
            return None

        return data.get(args[0], None) if len(args) == 1 \
            else get_value(data.get(args[0], {}), *args[1:])

    s, e = "<cbptools[\'", "\']>"
    if line.find(s) != -1:
        content = line[line.find(s) + len(s):line.find(e)]
        inplace, force = False, False

        if content.startswith('!'):
            content = content[1:]
            inplace = True

        elif content.startswith('+'):
            content = content[1:]
            force = True

        keys = content.split(':')
        value = get_value(config, *keys)

        if isinstance(value, dict):
            value = value if not value.get('file', None) else value.get('file')

        if keys[0] == 'input_data' and not value and not force:
            return False

        if inplace:
            line = line.replace('%s!%s%s' % (s, content, e), str(value))

        elif force:
            value = repr(value) if isinstance(value, str) else str(value)
            line = line.replace(
                '%s+%s%s' % (s, content, e),
                '%s = %s' % (keys[-1], value)
            )

        else:
            value = repr(value) if isinstance(value, str) else str(value)
            line = line.replace(
                '%s%s%s' % (s, content, e),
                '%s = %s' % (keys[-1], value)
            )

    return line


def get_participant_ids(file: str = None, sep: str = None,
                        index_col: str = 'participant_id') -> list:
    """Load the participants file and see if there are participants in it"""
    if file is None:
        logging.error('TypeError: [participants] Input is required')
        return []

    if not os.path.exists(file):
        logging.error('FileNotFoundError: [participants] No such file: %s'
                      % file)
        return []

    # Try to guess separator if none is given
    if sep is None:
        ext = os.path.splitext(file)[-1]
        separators = {'.tsv': '\t', '.csv': ','}
        if ext in separators.keys():
            sep = separators[ext]

    participants = list(pd.read_csv(
        file, sep=sep, engine='python'
    ).get(index_col, []))

    if not participants and sep is None:
        logging.error('ValueError: [participants] No participant indices '
                      'found in %s: %s.  Try adding a separator keyword to '
                      'the configuration file.' % (index_col, file))

    if not participants:
        logging.error('ValueError: [participants] No participant indices '
                      'found in %s: %s' % (index_col, file))
        return []

    return participants


def estimate_memory_usage(config: dict, masks: dict,
                          participants: list) -> dict:
    input_data_type = config.get('input_data_type')
    mem_mb = dict()

    # Connectivity task
    if input_data_type == 'rsfmri':
        buffer = 250  # in MB
        time_series = get_filepath(config['input_data']['time_series'])
        sizes = [
            os.path.getsize(time_series.format(participant_id=participant))
            for participant in participants
        ]
        mb_value = int(np.ceil(bytes_to(np.ceil(max(sizes)*2.5), 'mb')))
        mem_mb['connectivity'] = mb_value + buffer

    elif input_data_type == 'dmri':
        buffer = 250
        samples = config['input_data']['samples'] + '*'
        sizes = []
        for participant in participants:
            sizes.append(sum([
                os.path.getsize(sample)
                for sample in glob.glob(
                    samples.format(participant_id=participant)
                )])
            )
        mb_value = int(np.ceil(bytes_to(np.ceil(max(sizes)*2.5), 'mb')))
        mem_mb['connectivity'] = mb_value + buffer

    # Clustering task
    if input_data_type in ('rsfmri', 'dmri'):
        buffer = 250
        seed = np.count_nonzero(masks['seed_mask'].get_data())
        target = np.count_nonzero(masks['target_mask'].get_data())
        mb_value = bytes_to(seed * target * len(participants), 'mb')
        mb_value = int(np.ceil(mb_value))
        mem_mb['clustering'] = mb_value + buffer

    elif input_data_type == 'connectivity':
        buffer = 250
        connectivity = config['input_data']['connectivity_matrix']
        sizes = [
            os.path.getsize(connectivity.format(participant_id=participant))
            for participant in participants
        ]
        mb_value = int(np.ceil(bytes_to(np.ceil(max(sizes)), 'mb')))
        mem_mb['clustering'] = mb_value + buffer

    return mem_mb


def validate_paths(d: dict, input_type: str, data: dict,
                   participant_ids: list = None) -> dict:
    """Validate input file formatting and ensure files exist"""
    keys = [key for key, value in d.items()
            if input_type in value.get('input_type')]

    # Validate filepath format
    for key in keys:
        path = data.get(key, None)
        required = d.get(key, {}).get('required', False)
        template = d.get(key, {}).get('template', False)
        file_type = d.get(key, {}).get('file_type', None)
        expand = d.get(key, {}).get('expand', False)
        subs = d.get(key, {}).get('subs', False)

        if path is None and required:
            logging.error('TypeError: [%s] Input is required' % key)
            continue

        elif path is None and not required:
            logging.info('Missing optional input [%s]' % key)
            continue

        if subs and not isinstance(path, dict) and path:
            data[key] = {**{'file': path}, **{sub: None for sub in subs}}
            path = data[key]['file']

        elif subs and isinstance(path, dict):
            for sub in subs:
                if sub not in path.keys():
                    data[key][sub] = None

            path = data[key].get('file', None)

        if not isinstance(path, str):
            logging.error('TypeError: [%s] Input should be a text string'
                          % key)
            continue

        if not os.path.isabs(path):
            logging.error('TypeError: [%s] Input path should be absolute, '
                          'not relative' % key)

        if file_type is not None and \
                not any(path.endswith(ext) for ext in file_type):
            logging.error('TypeError: [%s] %s extension given, %s expected'
                          % (key, os.path.splitext(path)[-1], file_type))

        if template:
            format_str = string.Formatter().parse(path)
            if 'participant_id' not in \
                    [t[1] for t in format_str if t[1] is not None]:
                logging.error('TemplateError: [%s] Missing {participant_id} '
                              'template in %s' % (key, path))

        else:
            # Non-template files can be checked immediately
            if expand:
                if not (glob.glob(path + '*')):
                    logging.error('FileNotFoundError: [%s] No such file(s): %s'
                                  % (key, path))

            elif not os.path.exists(path):
                logging.error('FileNotFoundError: [%s] No such file or '
                              'directory: %s' % (key, path))

    # Ensure files are present for all participants
    if participant_ids is None:
        participant_ids = []

    bad_pids = []

    for pid in participant_ids:
        for key in keys:
            path = get_filepath(data.get(key, None))
            template = d.get(key, {}).get('template', False)
            expand = d.get(key, {}).get('expand', False)

            if template and path is not None:
                path = path.format(participant_id=pid)

                if expand:
                    if not (glob.glob(path + '*')):
                        logging.warning('FileNotFound: [%s] No such file(s): '
                                        '%s' % (key, path))
                        bad_pids.append(pid)

                elif not os.path.exists(path):
                    logging.warning('FileNotFound: [%s] No such file: '
                                    '%s' % (key, path))
                    bad_pids.append(pid)

    bad_pids = set(bad_pids)
    data = {key: data.get(key, None) for key in keys}
    data['participant_ids_bad'] = list(bad_pids)
    return data


def validate_time_series(time_series: str, participants: list,
                         seed_mask: spatialimage,
                         confounds: Union[str, dict] = None) -> list:
    """Assess whether the time-series are in the same space as the seed mask
    and whether confounds (if given) have matching timepoints to the
    time-series"""
    bad_pids = []
    for pid in participants:
        ts = time_series.format(participant_id=pid)
        img = nib.load(ts)

        # Check if time-series and seed mask are in the same space
        if not imgs_equal_3d(imgs=[img, seed_mask]):
            logging.warning('Mismatch: [time_series] %s and seed mask are not '
                            'in the same space' % ts)
            bad_pids.append(pid)

        # Check if all confounds columns are present
        if confounds:
            if isinstance(confounds, dict):
                cf = confounds.get('file').format(participant_id=pid)
                sep = confounds.get('sep', None)
                usecols = confounds.get('usecols', None)

                if sep is None:
                    ext = os.path.splitext(confounds.get('file'))[-1]
                    seps = {'.tsv': '\t', '.csv': ','}
                    sep = seps[ext] if ext in seps.keys() else None

                if usecols:
                    header = pd.read_csv(cf, sep=sep, header=None, nrows=1)
                    header = header.values.tolist()[0]
                    usecols = [x for x in header
                               if any(fnmatch(x, p) for p in usecols)]

                df = pd.read_csv(
                    cf, sep=sep, usecols=usecols, engine='python')

            else:
                cf = confounds.format(participant_id=pid)
                ext = os.path.splitext(cf)[-1]
                seps = {'.tsv': '\t', '.csv': ','}
                sep = seps[ext] if ext in seps.keys() else None
                df = pd.read_csv(cf, sep=sep, engine='python')

            if len(df) != img.shape[-1]:
                logging.warning('Mismatch: [confounds] %s and time-series '
                                'do not have matching timepoints' % cf)
                bad_pids.append(pid)

    bad_pids = list(set(bad_pids))
    return bad_pids


def validate_connectivity(connectivity_matrix: Union[str, dict],
                          participants: list,
                          seed_mask: spatialimage) -> list:
    """Assess whether connectivity matrices have the correct shape"""
    bad_pids = []
    n_voxels = np.count_nonzero(seed_mask.get_data())
    for pid in participants:
        file = get_filepath(connectivity_matrix)
        file = file.format(participant_id=pid)

        try:
            mat = np.load(file, mmap_mode='r')
            _, ext = os.path.splitext(file)

            if ext == '.npz':
                if 'connectivity' not in list(mat.keys()):
                    logging.warning('Cannot find connectivity.npy inside %s'
                                    % file)
                    bad_pids.append(pid)
                    continue

                mat = mat.get('connectivity')

        except:
            logging.warning('Unable to open %s' % file)
            bad_pids.append(pid)
            continue

        if mat.shape[0] != n_voxels:
            logging.warning('Mismatch: [connectivity] Expected shape '
                            '(%s, x), not (%s, x)' % (n_voxels, mat.shape[0]))
            bad_pids.append(pid)

    bad_pids = list(set(bad_pids))
    return bad_pids


def validate_parameters(d: dict, input_type: str, data: dict) -> dict:
    """Ensure required parameters are given and default values are used where
    none are entered. Returns a data dictionary object with only relevant
    parameters to the input_type."""
    for task in d.keys():
        keys = [key for key, value in d[task].items()
                if input_type in value.get('input_type')]

        for key in keys:
            if not data.get(task, None):
                data[task] = dict()

            d_task = d.get(task, {}).get(key, {})
            value = data.get(task, {}).get(key, None)
            required = d_task.get('required', False)
            instance_type = d_task.get('instance_type', False)
            default = d_task.get('default', None)
            allowed = d_task.get('allowed', None)

            if required and value is None:
                logging.error('TypeError: %s->%s requires a value.'
                              % (task, key))
                continue

            elif value is None and default is not None:
                logging.warning('DefaultValue: Setting %s->%s to %s'
                                % (task, key, default))
                data[task][key] = default
                continue

            if allowed is not None:
                if isinstance(value, list) and \
                        not set(value).issubset(allowed):
                    logging.error('InputError: [%s->%s] Must be %s, not %s'
                                  % (task, key, ', '.join(allowed), value))
                    continue

                elif isinstance(value, str) and value not in allowed:
                    logging.error('InputError: [%s->%s] Must be %s, not %s'
                                  % (task, key, ', '.join(allowed), value))
                    continue

            if type(value) is not locate(instance_type) and value is not None:
                logging.error('TypeError: [%s->%s] Must be %s, not %s'
                              % (task, key, instance_type,
                                 type(value).__name__))
                continue

        data[task] = {key: data.get(task, {}).get(key, None) for key in keys}

    # Ensure low_pass > high_pass for rsfmri data
    if input_type == 'rsfmri':
        low_pass = data.get('connectivity', {}).get('low_pass', None)
        high_pass = data.get('connectivity', {}).get('high_pass', None)
        tr = data.get('connectivity', {}).get('tr', None)

        if low_pass is not None and high_pass is not None:
            if high_pass >= low_pass:
                logging.error('ValueError: High-pass (%s) is expected to '
                              'be smaller than low-pass (%s)'
                              % (high_pass, low_pass))

            if tr is None:
                logging.error('ValueError: connectivity->tr requires a value')

            elif tr > 100:
                logging.warning('connectivity->tr is large. Are you sure it '
                                'is repetition time in seconds?')

    # Format values so that they can be used for FSL input directly
    elif input_type == 'dmri':
        task = 'connectivity'
        pd = data.get(task, {}).get('correct_path_distribution', False)
        loop_check = data.get(task, {}).get('loop_check', False)
        data[task]['correct_path_distribution'] = '--pd' if pd else ''
        data[task]['loop_check'] = '-l' if loop_check else ''

    return data


def load_img(name: str, mask: str) -> Union[spatialimage, bool]:
    """Check if the input mask can be read using nibabel"""
    try:
        return nib.load(mask)

    except Exception as e:
        logging.error('ValueError: [%s_mask] Unable to read contents of file: '
                      '%s' % (name, mask))
        return False


def process_seed_indices(config: dict) -> Union[dict, bool]:
    input_data = config.get('input_data', {})
    indices_file = get_filepath(input_data.get('seed_indices', None))
    seed = get_filepath(input_data.get('seed_mask', None))

    try:
        indices = np.load(indices_file, mmap_mode='r')

    except Exception as exc:
        logging.error('ValueError: [seed_indices] Unable to read contents of '
                      'file: %s' % indices_file)
        return False

    seed_img = load_img('seed', seed)
    n_voxels = np.count_nonzero(seed_img.get_data())

    if indices.shape != (n_voxels, 3):
        logging.error('ValueError: [seed_indices] Expected shape '
                      '(%s, 3), not %s'
                      % (n_voxels, str(indices.shape)))
        return False

    files = {'seed_mask': seed_img, 'seed_indices': indices}
    return files


def process_masks(config: dict) -> Union[dict, bool]:
    def base_proc(file, img, resample_to_mni: bool = False,
                  bin_threshold: float = 0.0):
        mni_mapped_voxels = map_voxels(
            voxel_size=[2, 2, 2],
            origin=[90, -126, -72],
            shape=(91, 109, 91)
        )
        data = img.get_data()

        # Binarize
        if not np.array_equal(data, data.astype(bool)):
            img = binarize_3d(img, threshold=bin_threshold)
            logging.warning('Binarizing %s: setting all values >%s to 1 and '
                            'others to 0' % (file, bin_threshold))

        # Resample
        if resample_to_mni:
            shape = img.shape
            affine = img.affine
            mapped_voxels = mni_mapped_voxels

            if not np.all([np.all(np.equal(a, b))
                           for a, b in zip(mapped_voxels, (shape, affine))]):
                img = resample_from_to(
                    img,
                    mapped_voxels,
                    order=0,
                    mode='nearest'
                )
                logging.warning('Resampling %s to MNI group template '
                                '(nibabel.processing.resample_from_to), '
                                'using order=0, mode=\'nearest\''
                                % file)

        return img

    masks = dict()

    # Input data
    input_data_type = config.get('input_data_type')
    input_data = config.get('input_data', {})
    seed = get_filepath(input_data.get('seed_mask', None))
    target = get_filepath(input_data.get('target_mask', None))

    # Parameters
    options = config.get('parameters', {}).get('masking', {})
    resample_to_mni = options['resample_to_mni']
    bin_threshold = options['threshold']
    median_filter = options['median_filter']
    median_filter_dist = options['median_filter_dist']

    # Load images
    seed_img = load_img('seed', seed)

    # Load default MNI152GM template if target is not given
    if target is None:
        target = 'Default MNI152 Gray Matter template'
        target_img = nib.load(
            pkg_resources.resource_filename(__name__,
                                            'templates/MNI152GM.nii.gz'))
        logging.warning('DefaultValue: [target_mask] Using default MNI152 '
                        'Gray Matter template')

    else:
        target_img = load_img('target', target)

    seed_img = base_proc(
        seed,
        seed_img,
        resample_to_mni=resample_to_mni,
        bin_threshold=bin_threshold
    )
    target_img = base_proc(
        target,
        target_img,
        resample_to_mni=resample_to_mni,
        bin_threshold=bin_threshold
    )

    if logging.error.count > 0:
        return False

    # Process seed
    if median_filter:
        seed_img = median_filter_img(seed_img, dist=median_filter_dist)
        logging.info('Applying median filter on seed (%s) (dist=%s)'
                     % (seed, median_filter_dist))

    masks['seed_mask'] = seed_img

    # Delete seed region from target
    del_seed_from_target = options['del_seed_from_target']
    del_seed_expand = options['del_seed_expand']

    if del_seed_from_target:
        target_img = subtract_img(
            source_img=target_img,
            target_img=seed_img,
            edge_dist=del_seed_expand
        )
        logging.info('Removing seed (%s) from target (%s) '
                     '(edge_dist=%s)' % (seed, target, del_seed_expand))

    if input_data_type == 'dmri':
        upsample = options['upsample_seed_to']

        if upsample is not None:
            if len(upsample) == 1:
                upsample = [upsample] * 3

            if len(upsample) != 3:
                logging.error('ValueError: [masking->upsample_seed_to] '
                              'Requires 1 or 3 values, not %s' % len(upsample))

            else:
                mapped_voxels = list(vox2out_vox(
                    (seed_img.shape, seed_img.affine),
                    upsample
                ))

                # Make sure affine signs are the same as the original
                a = np.sign(target_img.affine)
                b = np.sign(mapped_voxels[1])
                mapped_voxels[1] = mapped_voxels[1] * (a * b)

                highres_seed_img = stretch_img(
                    source_img=seed_img,
                    target=mapped_voxels
                )
                logging.info('Stretched seed (%s) to fit a '
                             '%s template with %s voxel size'
                             % (seed,
                                'x'.join(map(str, highres_seed_img.shape)),
                                'x'.join(map(str, upsample))))
                masks['highres_seed_mask'] = highres_seed_img

    # Process target
    if input_data_type == 'rsfmri':
        subsample = options['subsample']

        # Reduce the number of voxels in target
        if subsample:
            target_img = subsample_img(img=target_img, f=2)
            logging.info('Subsampling target image (%s)' % target)

    elif input_data_type == 'dmri':
        downsample = options['downsample_target_to']

        if downsample is not None:
            if len(downsample) == 1:
                downsample = [downsample] * 3

            if len(downsample) != 3:
                logging.error('ValueError: [masking->downsample_target_to] '
                              'Requires 1 or 3 values, not %s'
                              % len(downsample))

            else:
                mapped_voxels = list(vox2out_vox(
                    (target_img.shape, target_img.affine),
                    downsample
                ))

                # Make sure affine signs are the same as the original
                a = np.sign(target_img.affine)
                b = np.sign(mapped_voxels[1])
                mapped_voxels[1] = mapped_voxels[1] * (a * b)

                target_img = resample_from_to(
                    target_img,
                    mapped_voxels,
                    order=0,
                    mode='nearest'
                )
                logging.info('Resampling %s to %s voxel size '
                             '(nibabel.processing.resample_from_to), using '
                             'order=0, mode=\'nearest\''
                             % (target, 'x'.join(map(str, downsample))))

    if logging.error.count > 0:
        return False

    masks['target_mask'] = target_img

    return masks


def create_workflow(config: dict, mem_mb: dict, work_dir: str) -> None:
    input_data_type = config.get('input_data_type')
    templates = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'templates'
    )
    snakefiles = ['header.Snakefile', 'body.Snakefile']

    if input_data_type in ('rsfmri', 'dmri'):
        snakefiles.insert(1, '%s.Snakefile' % input_data_type)
        config['input_data']['connectivity_matrix'] = \
            'connectivity/connectivity_{participant_id}.npy'

    if input_data_type == 'rsfmri':
        config['input_data']['touchfile'] = 'log/.touchfile'

    config['mem_mb'] = mem_mb

    with open(os.path.join(work_dir, 'Snakefile'), 'w') as of:
        for snakefile in snakefiles:
            with open(os.path.join(templates, snakefile), 'r') as f:
                for line in f:
                    line = parse_line(line, config=config)

                    if line is not False:
                        of.write(line)

    shutil.copy(os.path.join(templates, 'cluster.json'), work_dir)


def create_project(work_dir: str, config: dict, masks: dict, mem_mb: dict,
                   participants: pd.DataFrame) -> dict:
    """Generate all the files needed by the CBP project"""
    input_data_type = config.get('input_data_type')

    # Save masks
    for key, value in masks.items():
        if isinstance(value, spatialimage):
            fpath = os.path.join(work_dir, key + '.nii')
            nib.save(value, fpath)
            logging.info('Created file %s' % fpath)

        elif isinstance(value, np.ndarray):
            fpath = os.path.join(work_dir, key + '.npy')
            np.save(fpath, value)
            logging.info('Created file %s' % fpath)

    # Save seed indices
    if input_data_type in ('rsfmri', 'dmri'):
        seed_indices = get_mask_indices(img=masks['seed_mask'], order='C')
        np.save(os.path.join(work_dir, 'seed_indices.npy'), seed_indices)
        config['input_data']['seed_indices'] = 'seed_indices.npy'

    # Save participant info
    n_bad = np.count_nonzero(participants.invalid)
    n_participants = participants.participant_id.count()
    if n_bad > 0:
        participants_bad = pd.DataFrame(
            participants[participants['invalid']]['participant_id']
        )
        participants_bad.to_csv(
            os.path.join(work_dir, 'participants_bad.tsv'),
            sep='\t',
            index=False
        )

    participants = pd.DataFrame(
        participants[~participants['invalid']]['participant_id']
    )
    participants.to_csv(
        os.path.join(work_dir, 'participants.tsv'),
        sep='\t',
        index=False
    )

    # Create workflow (snakefile)
    create_workflow(config=config, mem_mb=mem_mb, work_dir=work_dir)

    # Info
    n_inc = n_participants - n_bad
    logging.info('Removed participants: %s' % n_bad)
    logging.info('Included participants: %s' % n_inc)
    seed = np.count_nonzero(masks['seed_mask'].get_data())

    if input_data_type in ('rsfmri', 'dmri'):
        target = np.count_nonzero(masks['target_mask'].get_data())
        conn_size = seed * target * n_inc
        logging.info('Approximate size of all connectivity matrices: %s'
                     % readable_bytesize(conn_size, 8))

    clust_size = (seed * n_inc) + (seed * 2) + n_inc + 1
    logging.info('Approximate size of all cluster label files: %s'
                 % readable_bytesize(clust_size, 8))

    stats = {
        'n_bad_participants': n_bad,
        'n_participants_total': n_participants
    }

    return stats


def validate_config(configfile: str, work_dir: str, logfile: str,
                    verbose: bool = False):
    """Validate configuration file. Returns false if validation cannot
    continue"""
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

    if verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info('CBP tools version %s' % __version__)
    logging.info('Setup initiated on %s in environment %s'
                 % (time.strftime('%b %d %Y %H:%M:%S'), sys.prefix))

    try:
        # Sometimes username/hostname can't be found, it's not a big problem
        logging.info('Username of creator is \'%s\' with hostname \'%s\''
                     % (os.getlogin(), socket.gethostname()))
    except:
        pass

    # Load configfile
    with open(configfile, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error('Could not load configfile: %s' % exc)
            return False

    # Load predefined settings
    defaults = pkg_resources.resource_filename(__name__, 'schema.yaml')
    with open(defaults, 'r') as stream:
        defaults = yaml.safe_load(stream)

    # Validate input data
    input_data_type = config.get('input_data_type', None)

    if input_data_type not in defaults.get('input_data_types'):
        logging.error('TypeError: [input_data_type] Must be %s, not'
                      '%s' % (', '.join(defaults.get('input_data_types')),
                              input_data_type))
    else:
        input_data_type = config['input_data_type'] = input_data_type.lower()

    input_data = config.get('input_data', {})
    time_series = input_data.get('time_series', {})
    pids = []
    pids_bad = []

    eval_ts = time_series.get('validate_shape', False) \
        if isinstance(time_series, dict) else False

    if not input_data:
        logging.error('ValueError: [input_data] No %s input data found'
                      % input_data_type)

    else:
        if not isinstance(input_data.get('participants', None), dict):
            input_data['participants'] = {
                'file': input_data.get('participants', None),
                'sep': None,
                'index_col': 'participant_id'
            }

        pp_data = input_data.get('participants', {})
        pids = get_participant_ids(
            file=pp_data.get('file', None),
            sep=pp_data.get('sep', None),
            index_col=pp_data.get('index_col', 'participant_id')
        )
        input_data = validate_paths(
            d=defaults.get('input'),
            input_type=input_data_type,
            data=input_data,
            participant_ids=pids
        )

        pids_bad = set(input_data['participant_ids_bad'])

        if pids and len(set(pids) - pids_bad) <= 1:
            logging.error('ValueError: [participants] Not enough participants '
                          'left after removing those with missing or bad data')

    # Validate parameters
    parameters = config.get('parameters', {})
    if not parameters:
        logging.error('ValueError: [parameters] No %s parameters found'
                      % input_data_type)

    else:
        parameters = validate_parameters(
            d=defaults.get('parameters'),
            input_type=input_data_type,
            data=parameters
        )

    # Check if FSL is accessible
    if input_data_type == 'dmri':
        fsl = shutil.which('fsl')
        fsl_outputtype = os.getenv('FSLOUTPUTTYPE')
        fsl_dir = os.getenv('FSLDIR')
        probtrackx2 = shutil.which('probtrackx2')

        if probtrackx2:
            logging.info('FSLInfo: Executable path is \'%s\'' % fsl)
            logging.info('FSLInfo: probtrackx2 executable path is \'%s\''
                         % probtrackx2)
            logging.info('FSLInfo: Directory is %s' % fsl_dir)
            logging.info('FSLInfo: Output type is %s' % fsl_outputtype)

        else:
            logging.warning('ModuleNotFoundError: No module named '
                            '\'probtrackx2\' (FSL)')

    if logging.error.count > 0:
        return False

    config = OrderedDict({
        'input_data_type': input_data_type,
        'input_data': input_data,
        'parameters': parameters
    })

    # Process Masks
    masks = dict()
    if input_data_type in ('rsfmri', 'dmri'):
        masks = process_masks(config=config)

    elif input_data_type == 'connectivity':
        masks = process_seed_indices(config=config)

    if logging.error.count > 0:
        return False

    # Evaluate time-series
    if input_data_type == 'rsfmri' and eval_ts:
        pids_bad = list(pids_bad)
        pids_bad += validate_time_series(
            time_series=get_filepath(input_data['time_series']),
            participants=list(set(pids) - set(pids_bad)),
            seed_mask=masks.get('seed_mask', None),
            confounds=input_data.get('confounds', None)
        )

        pids_bad = set(pids_bad)

        if pids and len(set(pids) - pids_bad) <= 1:
            logging.error('ValueError: [participants] Not enough participants '
                          'left after removing those with missing or bad data')

    elif input_data_type == 'connectivity':
        pids_bad = list(pids_bad)
        pids_bad += validate_connectivity(
            connectivity_matrix=get_filepath(
                input_data['connectivity_matrix']),
            participants=list(set(pids) - set(pids_bad)),
            seed_mask=masks['seed_mask']
        )

        pids_bad = set(pids_bad)

        if pids and len(set(pids) - pids_bad) <= 1:
            logging.error('ValueError: [participants] Not enough participants '
                          'left after removing those with missing or bad data')

    if logging.error.count > 0:
        return False

    # Get estimated memory usage of tasks
    mem_mb = estimate_memory_usage(
        config=config,
        masks=masks,
        participants=list(set(pids) - set(pids_bad))
    )

    # Create the project files
    participants = pd.DataFrame(
        set(pids), columns=['participant_id']
    )
    participants.sort_values(by='participant_id', inplace=True)
    participants['invalid'] = np.where(
        participants['participant_id'].isin(pids_bad),
        True, False
    )
    info = create_project(
        work_dir=work_dir,
        config=config,
        masks=masks,
        mem_mb=mem_mb,
        participants=participants
    )
    logging.info('Project setup completed on %s'
                 % time.strftime('%b %d %Y %H:%M:%S'))
    logging.shutdown()
    return info


def copy_example(params, **kwargs):
    """Copy an example configuration file to the current working directory"""
    input_data_type = params.input_data_type
    filename = 'config_%s.yaml' % input_data_type
    file = pkg_resources.resource_filename(__name__, 'templates/%s' % filename)
    dest = os.path.join(os.getcwd(), filename)

    if os.path.exists(dest):
        path, ext = os.path.splitext(dest)
        dest = '%s ({i})%s' % (path, ext)
        i = 0
        while os.path.exists(dest.format(i=i)):
            i += 1

        dest = dest.format(i=i)

    shutil.copy(file, dest)
    print('Copied %s example configuration file to %s'
          % (input_data_type, dest))
