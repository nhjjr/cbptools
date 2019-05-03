#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  Create a project and its associated working directory
"""

from .version import __version__
from .utils import get_template, get_disk_size, CallCountDecorator, TColor
from .image import binarize_3d, stretch_img, median_filter_img, subtract_img, subsample_img, map_voxels
from .fslconf import FSL
from pydoc import locate
from typing import Any, Union, List, Tuple
from collections import OrderedDict
from nibabel.processing import resample_from_to, vox2out_vox
import nibabel as nib
import shutil
import numpy as np
import pandas as pd
import pkg_resources
import yaml
import os
import sys
import socket
import glob
import string
import logging
import time


class ProjectMessages:
    messages = {
        'FileReadError': 'Unable to read contents of file "{0}"',
        'FileNotFound': 'File(s) not found: {0}',
        'PathNotFound': 'Path not found: {1}',
        'FileCreated': 'Created file {0}',
        'FieldAdded': 'Added field "{0}" with value: "{1}"',
        'MissingInput': 'Missing {0} input for field {1}',
        'MissingData': 'Participant with id "{0}" removed due to missing data',
        'UnrecognizedField': 'Unrecognized {0} field: Input {1} to non-existing field {2}',
        'MissingTemplate': 'Template \{{0}\} not in {1}',
        'TypeError': 'Input type not recognized for {0}: {1} given, {2} expected',
        'InsufficientCount': 'Not enough {0}, {1} given, >={2} expected',
        'UsingDefault': 'Using default value ({0}) for {1}',
        'ModuleNotFound': 'Module not found: {0}',
        'SaveFailed': 'Save Failed: {0}',
        'ValueError': '{0} requires {1} values, but {2} were given'
    }

    def __init__(self):
        for k, v in self.messages.items():
            setattr(self, k, lambda *args, val=v: val.format(*args))


# Enable logging
E = ProjectMessages()


def pyyaml_ordereddict(dumper, data):
    value = []

    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)
        value.append((node_key, node_value))

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)


def remove_defaults(d):
    """Remove default values from d (that start with _) used only for verifying input"""
    if not isinstance(d, dict):
        return d

    elif 'value' in d.keys():
        # Either return the value as a string, or return multiple values as a 'key=value' list
        values = [f'{k}={v}' for k, v in d.items() if not k.startswith('_') and not k == 'value']
        return [d.get('value')] + values if values else d.get('value')

    return {k: v for k, v in ((k, remove_defaults(v)) for k, v in d.items())
            if not k.startswith('_') and (not isinstance(v, dict) or v.keys())}


if os.path.isfile('project.log'):
    os.remove('project.log')

logging.basicConfig(
    filename='project.log',
    format='%(asctime)s\t%(levelname)s\t%(message)s',
    datefmt='%I:%M:%S%p',
    level=logging.INFO
)
logging.error = CallCountDecorator(logging.error)
logging.warning = CallCountDecorator(logging.warning)

# make pyyaml work with OrderedDict
yaml.add_representer(OrderedDict, pyyaml_ordereddict)


class Participants(dict):
    """Store participant IDs"""
    def __init__(self, *args, **kwargs):
        super(Participants, self).__init__(*args, **kwargs)

    def read_csv(self, file: str, sep: str = None, index_col: str = 'participant_id') -> bool:
        if not os.path.exists(file):
            logging.error(E.FileReadError(file))
            return False

        participants = list(pd.read_csv(file, sep=sep).get(index_col, []))

        if not participants:
            logging.warning(E.MissingInput(index_col, file))
            return False

        self.add_iterable(participants)

    def add_iterable(self, participants):
        if hasattr(participants, '__iter__') and not isinstance(participants, str):
            for participant in participants:
                self.__dict__[participant] = True

        else:
            logging.warning(E.TypeError('participants', type(participants).__name__, 'iterable'))

    def count(self, only_valid: bool=True):
        return sum(1 for k in self.__dict__.keys() if self.__dict__[k]) if only_valid else len(self.__dict__)

    def set_invalid(self, participant_id: str):
        self.__dict__[participant_id] = False
        logging.warning(E.MissingData(participant_id))

    def tolist(self, only_valid: bool=True):
        return [k for k, v in self.__dict__.items() if v] if only_valid else list(self.__dict__.keys())

    def save(self, work_dir) -> None:
        """Save valid participant indices in tsv format"""
        participants = pd.DataFrame({'participant_id': self.tolist()})
        participants.to_csv(os.path.join(work_dir, 'participants.tsv'), sep='\t', index=False)
        logging.info(E.FileCreated(os.path.join(work_dir, 'participants.tsv')))


class DataSet(dict):
    """This class will contain all the files for the dataset, and that includes participants as well."""

    def __init__(self, *args, **kwargs):
        super(DataSet, self).__init__(*args, **kwargs)

        with open(pkg_resources.resource_filename(__name__, 'dataset.yaml'), 'r') as stream:
            self.update(yaml.load(stream))

        self['_participants'] = Participants()
        self['_base_path'] = ''

    def add(self, index: str, **kwargs: Any) -> None:
        if index in self.keys():
            for key, value in kwargs.items():
                if key in ['value'] + self[index].get('_keys', []):
                    self[index][key] = value
                else:
                    logging.warning(E.UnrecognizedField('file', value, f'{index}->{key}'))

        elif index in ('base_path', 'dataset', 'dataset_path'):
            self['_base_path'] = kwargs.get('value')

        else:
            logging.warning(E.UnrecognizedField('file', kwargs.get('value'), index))

    def get_value(self, index: str, key: str = 'value') -> Any:
        return self.get(index, {}).get(key, None)

    def participants(self) -> Participants:
        return self['_participants']

    def set_participants(self, participants: Union[str, List[str], Tuple[str]], **kwargs) -> bool:
        if type(participants) is str:
            if not os.path.exists(participants):
                logging.warning(E.FileNotFound(participants))
                return False

            self['_participants'].read_csv(participants, **kwargs)
            return True

        elif type(participants) in [list, tuple]:
            if not all(isinstance(participant, str) for participant in participants):
                logging.warning(E.TypeError(
                    'participants',
                    ', '.join(set([type(participant).__name__ for participant in participants])),
                    'str'))
                return False

            self['_participants'].add_iterable(participants)
            return True

        else:
            logging.warning(E.TypeError('participants', type(participants).__name__, 'file or list'))
            return False

    def _file_exists(self, index, participant_id) -> bool:
        value = self.get(index).get('value', None).format(participant_id=participant_id)

        if self.get(index).get('_expand', False):
            # Input is not a file, but an expansion path
            if not (glob.glob(value + '*')):
                logging.warning(E.FileNotFound(value + '*'))
                return False

        elif not os.path.exists(value):
            logging.warning(E.FileNotFound(value))
            return False

        return True

    def _is_valid(self, index) -> bool:
        """Check if the requested entry is valid, not if the file exists"""
        file = self.get(index, {})
        is_required = file.get('_required', False)
        allowed_extensions = file.get('_extension', None)
        value = file.get('value', None)

        # Verify if item is required and has a value -- if not required and has no value, pass True
        if is_required and value is None:
            logging.error(E.MissingInput('file', index))
            return False

        elif value is None:
            return True

        # Check if the file extension matches the requirements
        if allowed_extensions is not None and not any(value.endswith(ext) for ext in allowed_extensions):
            logging.error(E.TypeError(index, os.path.splitext(value)[-1], allowed_extensions))
            return False

        # Check if {participant_id} wildcard is present in the file
        format_args = [t[1] for t in string.Formatter().parse(value) if t[1] is not None]

        if 'participant_id' not in format_args:
            logging.error(E.MissingTemplate('participant_id', file))
            return False

        return True

    def validate(self, modality: str) -> bool:
        if self['_participants'].count() <= 1:
            logging.error(E.InsufficientCount('participants', self['_participants'].count(), 2))
            return False

        # Check if base_path is a valid directory
        if self['_base_path'] != '' and not os.path.isdir(self['_base_path']):
            logging.error(E.PathNotFound(self["_base_path"]))
            return False

        # Loop through all files except those starting with _ (_participants, _base_path)
        for file in (x for x in self.keys() if not x.startswith('_')):
            if self[file].get('_modality', 'all') not in ('all', modality):
                continue

            # Prepend base_path to dataset files
            self[file]['value'] = os.path.join(self['_base_path'], self.get(file).get('value', None))

            if not self._is_valid(file):  # this method adds an error already
                continue

            # Only select valid participants -- only one missing file is sufficient to remove them
            participants = self['_participants'].tolist()

            for participant in participants:
                if not self._file_exists(file, participant):
                    self['_participants'].set_invalid(participant)

        if self['_participants'].count() <= 1:
            logging.error(E.InsufficientCount('participants', self['_participants'].count(), 2))

    def save(self, modality) -> dict:
        """Present all relevant values as a dict to store into a yaml file"""
        # Remove all files that are not necessary for the chosen modality
        for k, v in self.items():
            if not k.startswith('_') and v.get('_modality', 'all') not in ('all', modality):
                self[k] = {}

        dataset = remove_defaults(self)

        # Log all parameters that are added
        for k, v in dataset.items():
            v = ', '.join(str(x) for x in v).replace('\t', '\\t') if isinstance(v, (list, tuple)) else v
            logging.info(E.FieldAdded(k, v))

        return dataset


class Parameters(dict):
    def __init__(self, *args, **kwargs):
        super(Parameters, self).__init__(*args, **kwargs)

        with open(pkg_resources.resource_filename(__name__, 'defaults.yaml'), 'r') as stream:
            self.update(yaml.load(stream))

    def add(self, task: str, param: str, value: Any) -> None:
        if task in self.keys() and param in self[task].keys():
            self[task][param]['value'] = value

        else:
            logging.warning(E.UnrecognizedField('parameter', value, '.'.join([task, param])))

    def get_value(self, task: str, param: str) -> Any:
        return self.get(task, {}).get(param, {}).get('value', None)

    def get_task(self, task: str) -> Any:
        task_values = dict()
        for k, v in self.get(task, {}).items():
            task_values[k] = v.get('value', None)

        return task_values

    def _is_valid(self, task, param) -> bool:
        key = '.'.join([task, param])
        parameter = self.get(task, {}).get(param, None)
        value = self.get_value(task, param)
        required = parameter.get('_required', False)
        dtype = parameter.get('_dtype')
        default = parameter.get('_default', None)
        allowed = parameter.get('_allowed', None)

        if required and not value:
            logging.error(E.MissingInput('parameter', key))
            return False

        elif value is None:
            if default is not None:
                self[task][param]['value'] = default
                logging.warning(E.UsingDefault(default, key))

            return True

        if allowed is not None:
            if isinstance(value, list) and not set(value).issubset(allowed):
                logging.error(E.TypeError(key, ', '.join(list(set(value) - set(allowed))), allowed))
                return False

            elif isinstance(value, str) and value not in allowed:
                logging.error(E.TypeError(key, value, allowed))
                return False

        if type(value) is not locate(dtype) and value is not None:
            logging.error(E.TypeError(key, type(value).__name__, dtype))
            return False

        return True

    def validate(self, modality: str):
        for task in self.keys():
            for param, values in self[task].items():
                if values.get('_modality') in ['all', modality]:
                    if not self._is_valid(task, param):
                        continue

        # Specific parameter restrictions
        low_pass = self.get_value('connectivity', 'low_pass')
        high_pass = self.get_value('connectivity', 'high_pass')
        tr = self.get_value('connectivity', 'tr')

        if low_pass is not None and high_pass is not None:
            if high_pass >= low_pass:
                logging.error(f'High-pass ({high_pass}) should be smaller than low-pass ({low_pass})')

            if tr is None:
                logging.error('High- and low-pass filters are set, but TR is missing')

    def save(self, modality) -> dict:
        """Present all relevant values as a dict to store into a yaml file"""
        def remove_defaults(d):
            """Remove default values from d (that start with _) used only for verifying input"""
            if not isinstance(d, dict):
                return d

            elif 'value' in d.keys():
                # Either return the value as a string, or return multiple values as a 'key=value' list
                values = [f'{k}={v}' for k, v in d.items() if not k.startswith('_') and not k == 'value']
                return [d.get('value')] + values if values else d.get('value')

            return {k: v for k, v in ((k, remove_defaults(v)) for k, v in d.items())
                    if not k.startswith('_') and (not isinstance(v, dict) or v.keys())}

        # Remove all parameters that are not necessary for the chosen modality
        for task in self.keys():
            for k, v in self[task].items():
                if not k.startswith('_') and v.get('_modality', 'all') not in ('all', modality):
                    self[task][k] = {}

        parameters = remove_defaults(self)

        # Log all parameters that are added
        for task in parameters.keys():
            for k, v in parameters[task].items():
                v = ', '.join(str(x) for x in v).replace('\t', '\\t') if isinstance(v, (list, tuple)) else v
                logging.info(E.FieldAdded('.'.join([task, k]), v))

        return parameters


class Masks:
    mni_mapped_voxels = map_voxels(
        voxel_size=[2, 2, 2],
        origin=[90, -126, -72],
        shape=(91, 109, 91)
    )

    def __init__(self):
        self.imgs = {
            'seed': {'file': None, 'img': None, 'high-res': None},
            'target': {'file': None, 'img': None}
        }
        self.modality = None
        self.options = None

    def add(self, index: str, file: str):
        if index in self.imgs.keys():
            self.imgs[index]['file'] = file
        else:
            logging.warning(E.UnrecognizedField('mask', file, index))

    def is_valid(self, index):
        file = self.imgs[index]['file']

        if file is None:
            logging.error(E.MissingInput('mask image', index))
            return False

        if not os.path.exists(file):
            logging.error(E.FileNotFound(index))
            return False

        try:
            self.imgs[index]['img'] = nib.load(file)

        except Exception as e:
            logging.error(E.FileReadError(file))
            return False

        return True

    def validate(self):
        if all([self.is_valid('seed'), self.is_valid('target')]):
            return True
        else:
            return False

    def _base_processing(self, index):
        data = self.imgs[index]['img'].get_data()
        basename = os.path.basename(self.imgs[index]["file"])
        resample_to_mni = self.options.get('resample_to_mni', True)
        bin_threshold = self.options.get('threshold', 0.0)

        # Binarize
        if not np.array_equal(data, data.astype(bool)):
            self.imgs[index]['img'] = binarize_3d(self.imgs[index]['img'], threshold=bin_threshold)
            logging.warning(f'Binarizing {basename}: setting all values >{bin_threshold} to 1 and others to 0')

        # Resample
        if resample_to_mni:
            shape = self.imgs[index]['img'].shape
            affine = self.imgs[index]['img'].affine
            mapped_voxels = self.mni_mapped_voxels  # TODO: Check if this can change based on input
            basename = os.path.basename(self[index]["file"])

            if not np.all([np.all(np.equal(a, b)) for a, b in zip(mapped_voxels, (shape, affine))]):
                self.imgs[index]['img'] = resample_from_to(
                    self.imgs[index]['img'],
                    mapped_voxels,
                    order=0,
                    mode='nearest'
                )
                logging.warning(f'Resampling {basename} to MNI group template '
                                f'(nibabel.processing.resample_from_to), using order=0, mode=\'nearest\'')

    def _process_seed(self):
        median_filter = self.options.get('median_filter', False)
        median_filter_dist = self.options.get('median_filter_dist', 1)
        basename = os.path.basename(os.path.basename(self.imgs["seed"]["file"]))

        if median_filter:
            self.imgs['seed']['img'] = median_filter_img(self.imgs['seed']['img'], dist=median_filter_dist)
            logging.info(f'Applying median filter on seed ({basename}) (dist={median_filter_dist})')

        if self.modality == 'dmri':
            upsample = self.options.get('upsample_seed_to', None)

            if upsample is not None:
                if len(upsample) == 1:
                    upsample = [upsample] * 3
                elif len(upsample) != 3:
                    logging.error(E.ValueError('masking.upsample_seed_to', '1 or 3', len(upsample)))
                    return

                mapped_voxels = (self.imgs['seed']['img'].shape, self.imgs['seed']['img'].affine)
                self.imgs['seed']['high-res'] = stretch_img(
                    source_img=self.imgs['seed']['img'],
                    target=vox2out_vox(mapped_voxels, upsample)
                )
                logging.info(f'Stretched seed ({basename}) to fit a '
                             f'{"x".join(map(str, self.imgs["seed"]["high-res"].shape))} template with '
                             f'{"x".join(map(str, upsample))} voxel size')

    def _process_target(self):
        basename = os.path.basename(self.imgs['target']['file'])

        if self.modality == 'fmri':
            del_seed_from_target = self.options.get('del_seed_from_target', False)
            del_seed_dist = self.options.get('del_seed_dist', 0)
            subsample = self.options.get('subsample', False)

            # Remove seed voxels from target
            if del_seed_from_target:
                seed_basename = os.path.basename(self.imgs["seed"]["file"])
                self.imgs['target']['img'] = subtract_img(
                    source_img=self.imgs['target']['img'],
                    target_img=self.imgs['seed']['img'],
                    edge_dist=del_seed_dist
                )
                logging.info(f'Removing seed ({seed_basename}) from target ({basename}) (edge_dist={del_seed_dist}')

            # Reduce the number of voxels in target
            if subsample:
                self.imgs['target']['img'] = subsample_img(img=self.imgs['target']['img'], f=2)
                logging.info(f'Subsampling target image ({basename})')

        elif self.modality == 'dmri':
            downsample = self.options.get('downsample_target_to', None)

            if downsample is not None:
                if len(downsample) == 1:
                    downsample = [downsample] * 3
                elif len(downsample) != 3:
                    logging.error(E.ValueError('masking.downsample_target_to', '1 or 3', len(downsample)))
                    return

                mapped_voxels = vox2out_vox(
                    (self.imgs['target']['img'].shape, self.imgs['target']['img'].affine),
                    downsample
                )

                self.imgs['target']['img'] = resample_from_to(
                    self.imgs['target']['img'],
                    mapped_voxels,
                    order=0,
                    mode='nearest'
                )
                logging.warning(f'Resampling {basename} to {"x".join(map(str, downsample))} voxel size '
                                f'(nibabel.processing.resample_from_to), using order=0, mode=\'nearest\'')

    def get_files(self):
        return {k: v['file'] for k, v in self.imgs.items() if v['file'] is not None}

    def save(self, work_dir: str, modality: str, **options):
        self.modality = modality
        self.options = options

        if not self.validate():
            return

        self._process_seed()
        self._process_target()

        nib.save(self.imgs['seed']['img'], os.path.join(work_dir, 'seed_mask.nii'))
        logging.info(E.FileCreated(os.path.join(work_dir, 'seed_mask.nii')))

        nib.save(self.imgs['target']['img'], os.path.join(work_dir, 'target_mask.nii'))
        logging.info(E.FileCreated(os.path.join(work_dir, "target_mask.nii")))

        if self.modality == 'dmri':  # TODO: Need to check if upsampled
            nib.save(self.imgs['seed']['high-res'], os.path.join(work_dir, 'highres_seed_mask.nii'))
            logging.info(E.FileCreated(os.path.join(work_dir, "highres_seed_mask.nii")))


class Project:
    def __init__(self):
        self.modality = None
        self.force_overwrite = False
        self.work_dir = None
        self.dataset = DataSet()
        self.parameters = Parameters()
        self.masks = Masks()
        self._modalities = ('fmri', 'dmri')
        self._tasks = tuple(self.parameters.keys())  # TODO: only validate up to what task is required

        # Logging header
        logging.info(f'CBP tools version {__version__}')
        logging.info(f'Setup initiated on {time.strftime("%b %d %Y %H:%M:%S")} in environment {sys.prefix}')
        logging.info(f'Username of creator is "{os.getlogin()}" with hostname "{socket.gethostname()}"')

    def add_mask(self, index: str, value: str) -> None:
        if index == 'region_of_interest':
            index = 'seed'

        self.masks.add(index, value)

    def add_file(self, index: str, value: Any, **kwargs) -> None:
        self.dataset.add(index, value=value, **kwargs)

    def add_path(self, index: str, value: Any, **kwargs) -> None:
        self.add_file(index, value=value, **kwargs)

    def add_parameter(self, task: str, param: str, value: Any) -> None:
        self.parameters.add(task, param, value)

    def add_param(self, task: str, param: str, value: Any) -> None:
        self.add_parameter(task, param, value)

    def set_participants(self, participants: Union[str, List[str], Tuple[str]], **kwargs) -> None:
        self.dataset.set_participants(participants, **kwargs)

    def _usage_info(self) -> None:
        """Add disk space requirements for some output files to the log"""
        roi_voxels = (np.asarray(nib.load(os.path.join(self.work_dir, 'seed_mask.nii')).get_data()) == 1).sum()
        target_voxels = (np.asarray(nib.load(os.path.join(self.work_dir, 'target_mask.nii')).get_data()) == 1).sum()
        n_participants = self.dataset.participants().count()
        connectivity_size = roi_voxels*target_voxels*n_participants
        cluster_labels_size = (roi_voxels*n_participants) + (roi_voxels*2) + n_participants + 1

        logging.info(f'Approximate size for connectivity results: {get_disk_size(connectivity_size, 8)}')
        logging.info(f'Approximate size for clustering results: {get_disk_size(cluster_labels_size, 8)}')
        logging.info(f'Size of summary results cannot be approximated')

    def _validate(self):
        """Validate all fields"""
        if self.modality not in self._modalities:
            logging.error(E.TypeError('modality', self.modality, ', '.join(self._modalities)))

        if self.work_dir is None:
            logging.error(E.MissingInput('project', 'work_dir (define using Project.work_dir = "path/to/dir")'))

        self.parameters.validate(self.modality)
        self.dataset.validate(self.modality)
        self.masks.validate()

        if self.modality == 'dmri':
            fsl = FSL()
            if fsl.has_probtrackx2():
                logging.info(f'FSL executable path is \'{fsl.fsl}\'')
                logging.info(f'probtrackx2 executable path is \'{fsl.probtrackx2}\'')
                logging.info(f'FSL directory: {fsl.fsl_dir}')
                logging.info(f'FSL output type: {fsl.fsl_outputtype}')

            else:
                logging.error(E.ModuleNotFound('probtrackx2'))

        if logging.error.count > 0:
            return False

        return True

    def save(self):
        def summary(msg: str, n_bad_participants: int=None, log_dir: str=os.getcwd()) -> None:
            if logging.error.count > 0:
                print(TColor.FAIL + msg + TColor.ENDC)
                print(TColor.OKBLUE + 'Log file:' + TColor.ENDC + f'{os.path.join(log_dir, "project.log")}')
                print(TColor.FAIL + f'{logging.error.count} errors in project' + TColor.ENDC)

            else:
                print(TColor.OKGREEN + msg + TColor.ENDC)
                print(TColor.OKBLUE + 'Log file:' + TColor.ENDC + f'{os.path.join(log_dir, "project.log")}')
                print(f'{logging.error.count} errors in project')

            if logging.warning.count > 0:
                print(TColor.WARNING + f'{logging.warning.count} warnings in project' + TColor.ENDC)
            else:
                print(f'{logging.warning.count} warnings in project')

            if n_bad_participants is not None and n_bad_participants > 0:
                print(TColor.WARNING + f'{n_bad_participants} participant(s) removed due to missing data' + TColor.ENDC)

        if not self._validate():
            n_bad = self.dataset.participants().count(only_valid=False) - self.dataset.participants().count()
            summary(E.SaveFailed('Resolve all errors before continuing'), n_bad)
            return

        n_bad = self.dataset.participants().count(only_valid=False) - self.dataset.participants().count()

        # Create Project directory
        if os.path.exists(self.work_dir) and len(os.listdir(self.work_dir)) > 0 and not self.force_overwrite:
            summary(E.SaveFailed('work_dir already exists and force_overwrite is False'), n_bad)
            return

        if not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)
            if not os.path.isdir(self.work_dir):
                summary(E.SaveFailed(f'work_dir ({self.work_dir}) could not be created'), n_bad)
                return

        if not os.path.isdir(os.path.join(self.work_dir, 'log')):
            os.makedirs(os.path.join(self.work_dir, 'log'))
            if not os.path.isdir(os.path.join(self.work_dir, 'log')):
                summary(E.SaveFailed(f'log dir ({os.path.join(self.work_dir, "log")}) could not be created'), n_bad)
                return

        # Move scripts & snakemake file to project directory
        project_files = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'project_files')

        if os.path.exists(os.path.join(self.work_dir, 'scripts')):
            shutil.rmtree(os.path.join(self.work_dir, 'scripts'))

        shutil.copy(os.path.join(project_files, 'Snakefile'), self.work_dir)
        shutil.copy(os.path.join(project_files, 'cluster.json'), self.work_dir)
        shutil.copytree(os.path.join(project_files, 'scripts'), os.path.join(self.work_dir, 'scripts'))

        self.dataset.participants().save(self.work_dir)
        self.masks.save(self.work_dir, self.modality, **self.parameters.get_task('masking'))

        with open(os.path.join(self.work_dir, 'project.yaml'), 'w') as stream:
            yaml.dump(OrderedDict({
                'project': {'work_dir': self.work_dir, 'modality': self.modality},
                'dataset': self.dataset.save(self.modality),
                'masks': self.masks.get_files(),
                'parameters': self.parameters.save(self.modality),
                'participants': sorted(self.dataset.participants().tolist(only_valid=True))
            }), stream, default_flow_style=False)

        self._usage_info()  # add approximate disk space requirements to log file
        logging.info(f'Project setup completed on {time.strftime("%b %d %Y %H:%M:%S")}')
        logging.shutdown()  # no more log entries are made
        shutil.move(os.path.join(os.getcwd(), 'project.log'), os.path.join(self.work_dir, 'log', 'project.log'))
        summary(f'New project created in {self.work_dir}', n_bad_participants=n_bad,
                log_dir=os.path.join(self.work_dir, 'log'))
        print(f'Manually edit {os.path.join(self.work_dir, "cluster.json")} to execute the workflow on a cluster '
              f'(e.g., SLURM or qsub)')
