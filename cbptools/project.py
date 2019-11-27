from .utils import (readable_bytesize, CallCountDecorator, TColor, bytes_to,
                    npy_header, npz_headers)
from .image import imgs_equal_3d, extract_regions
from .workflow import build_workflow
from nibabel.filebasedimages import ImageFileError
from nibabel.spatialimages import SpatialImage
from pandas.io.common import EmptyDataError
from typing import Union, Tuple
import nibabel as nib
import pandas as pd
import numpy as np
import pkg_resources
import itertools
import logging
import fnmatch
import shutil
import yaml
import glob
import os

logging.error = CallCountDecorator(logging.error)
logging.warning = CallCountDecorator(logging.warning)


class MaskError(Exception):
    """Raised when an input mask fails validation"""
    pass


class SilentError(Exception):
    """Error without message in case the message has already been logged"""
    pass


class DataSet:
    def __init__(self, document: dict):
        self.modality = document.get('modality', None)
        self.data = document.get('data', {})
        self.template = pkg_resources.resource_filename(
            __name__, 'templates/MNI152GM.nii.gz')
        self.seed_coordinates = None
        self.references = None
        self.mem_mb = {'connectivity': 0, 'clustering': 0}
        self.ppids = []
        self.ppids_bad = []
        self.n_seed_voxels = 0
        self.n_target_voxels = 0

        # Padding for memory estimate in MB
        self.mem_padding = 250

    @staticmethod
    def get_ppids(file: str, delimiter: str = None,
                  index_column: str = None) -> list:
        """Retrieve participant ids from participant file"""
        if not os.path.exists(file):
            logging.error('No such file: %s' % file)
            return []

        df = pd.read_csv(file, sep=delimiter, engine='python')

        if delimiter is None:
            sep = df._engine.data.dialect.delimiter
            logging.warning('no delimiter set, inferred delimiter is %s' % sep)

        if index_column is None:
            index = 'participant_id'
            logging.warning('index_column is undefined, using %s' % index)

        if index_column not in df.columns:
            logging.error('index_column %s not found in %s'
                          % (index_column, file))
            return []

        ppids = list(df.get(index_column))

        if len(ppids) == 0:
            logging.error('no subject-ids found in %s' % file)

        return list(set(ppids))

    @staticmethod
    def load_img(file: str, level: str = None) -> Union[SpatialImage, None]:
        """Attempt to load a NIfTI image"""
        try:
            return nib.load(file)
        except (ImageFileError, FileNotFoundError) as exc:
            if level == 'error':
                logging.error(exc)
            elif level == 'warning':
                logging.warning(exc)
            return None

    def _masks(self, seed: str, target: str,
               space: str) -> Union[Tuple[SpatialImage, SpatialImage], None]:
        """Load and validate standard-space input masks"""
        level = 'warning' if space == 'native' else 'error'
        seed_img = self.load_img(seed, level=level)

        # Check if seed_img may be an atlas
        region_id = self.data['masks'].get('region_id', None)
        if region_id and seed_img is not None:
            try:
                seed_img = extract_regions(seed_img, region_id)
            except ValueError as exc:
                raise MaskError(exc)

        if target is None and space == 'standard':
            logging.warning('using default target: 2mm^3 MNI152 gray matter')
            target_img = self.load_img(self.template, level=level)
        else:
            target_img = self.load_img(target, level=level)

        # Check if masks were loaded
        if seed_img is None or target_img is None:
            raise SilentError()

        # Check if masks are in the same space
        if not imgs_equal_3d([seed_img, target_img]):
            raise MaskError('seed- and target-mask are not in the same space')

        # Ensure that masks have no weird values and can be binarized
        masks = (('seed', seed, seed_img), ('target', target, target_img))
        for name, file, img in masks:
            data = img.get_data()
            n_voxels = np.count_nonzero(data)

            # Get maximum seed/target size for size/memory estimation
            if name == 'seed' and n_voxels > self.n_seed_voxels:
                self.n_seed_voxels = n_voxels

            elif name == 'target' and n_voxels > self.n_target_voxels:
                self.n_target_voxels = n_voxels

            # Assess validity of input mask
            if not np.array_equal(data, data.astype(bool)):
                logging.warning('%s is not a binary mask file' % file)

                n_nans = np.count_nonzero(np.isnan(data))
                n_infs = np.count_nonzero(np.isinf(data))

                if n_nans > 0:
                    raise MaskError('%s NaN values in %s. Please manually'
                                    'verify %s' % (n_nans, name, file))

                if n_infs > 0:
                    raise MaskError('%s inf values in %s. Please manually'
                                    'verify %s' % (n_nans, name, file))

        return seed_img, target_img

    def _rsfmri(self) -> bool:
        """Validate rsfMRI input data"""
        seed_file = self.data['masks']['seed']
        target_file = self.data['masks'].get('target', None)
        space = self.data['masks']['space']
        ts_file = self.data['time_series']
        confounds = self.data.get('confounds', {})
        sessions = self.data.get('session', [None])

        # Prepare confounds
        if confounds.get('file', None):
            if confounds.get('delimiter', None) is None:
                confounds['delimiter'] = '\t'
                logging.warning('no confounds delimiter, using default: \\t')

            if confounds.get('columns', None) is None:
                logging.warning('no confounds columns defined, using all '
                                'columns')

        # Prepare seed and target if standard space is used
        if space == 'standard':
            try:
                seed_img, target_img = self._masks(
                    seed_file, target_file, space)
            except MaskError as exc:
                logging.error(exc)
                return False
            except SilentError:
                return False

        # Validate subject-specific files
        for ppid, session in itertools.product(self.ppids, sessions):
            if session:
                name = 'subject-id %s, session %s' % (ppid, session)
            else:
                name = 'subject-id %s' % ppid

            # Check if time-series file exists
            ts = ts_file.format(participant_id=ppid, session=session)
            ts_img = self.load_img(ts)
            if ts_img is None:
                logging.warning('missing expected file: %s' % ts)
                self.ppids_bad.append(ppid)
                continue

            # Validate native masks
            if space == 'native':
                ppid_seed = seed_file.format(participant_id=ppid)
                ppid_target = target_file.format(participant_id=ppid)

                try:
                    seed_img, target_img = self._masks(
                        ppid_seed, ppid_target, space)
                except MaskError as exc:
                    logging.warning(exc)
                    self.ppids_bad.append(ppid)
                    continue
                except SilentError:
                    self.ppids_bad.append(ppid)
                    continue

            # Check if time-series and masks are in the same space
            if not imgs_equal_3d([ts_img, seed_img, target_img]):
                logging.warning('time_series and masks are not in the '
                                'same space for %s' % name)
                self.ppids_bad.append(ppid)
                continue

            # Check if confounds are valid
            if confounds.get('file', None):
                try:
                    df = pd.read_csv(
                        confounds['file'].format(
                            participant_id=ppid,
                            session=session
                        ),
                        sep=confounds['delimiter'],
                        engine='python'
                    )
                except (EmptyDataError, FileNotFoundError) as exc:
                    logging.warning(exc)
                    self.ppids_bad.append(ppid)
                    continue

                if confounds.get('columns', None):
                    header = df.columns.tolist()
                    usecols = []
                    for col in confounds.get('columns', None):
                        if '*' in col:
                            usecols += [x for x in header if any(
                                fnmatch.fnmatch(x, p) for p in [col])]
                        else:
                            usecols.append(col)

                    usecols = [i for i in usecols if i is not None]

                    if not set(usecols).issubset(set(header)):
                        missing = list(set(usecols) - set(header))
                        logging.warning(
                            'missing confounds columns for %s: %s'
                            % (name, ', '.join(missing)))
                        self.ppids_bad.append(ppid)
                        continue

                if len(df) != ts_img.shape[-1]:
                    logging.warning('confounds and time-series for %s do not '
                                    'have matching timepoints' % name)
                    self.ppids_bad.append(ppid)
                    continue

        self.ppids_bad = list(set(self.ppids_bad))
        self.ppids = list(set(self.ppids) - set(self.ppids_bad))
        return True

    def _dmri(self) -> bool:
        """Validate dMRI input data"""
        seed_file = self.data['masks']['seed']
        target_file = self.data['masks'].get('target', None)
        space = self.data['masks']['space']
        bet_binary_mask_file = self.data['bet_binary_mask']
        xfm_file = self.data['xfm']
        inv_xfm_file = self.data['inv_xfm']
        samples = self.data['samples']
        sessions = self.data.get('session', [None])

        # Prepare seed and target if standard space is used
        if space == 'standard':
            try:
                _, _ = self._masks(seed_file, target_file, space)
            except MaskError as exc:
                if exc is not None:
                    logging.error(exc)
                return False

        for ppid, session in itertools.product(self.ppids, sessions):
            if session:
                name = 'subject-id %s, session %s' % (ppid, session)
            else:
                name = 'subject-id %s' % ppid

            # Validate native masks
            if space == 'native':
                ppid_seed = seed_file.format(participant_id=ppid)
                ppid_target = target_file.format(participant_id=ppid)

                try:
                    _, _ = self._masks(ppid_seed, ppid_target, space)
                except MaskError as exc:
                    if exc is not None:
                        logging.warning(exc)
                    self.ppids_bad.append(ppid)
                    continue

            # Validate merged samples
            merged = glob.glob(samples.format(
                participant_id=ppid, session=session) + '*')

            if not merged:
                logging.warning('required file(s) is/are missing for %s'
                                % name)
                self.ppids_bad.append(ppid)
                continue

            # Validate bet_binary_mask
            for file in [bet_binary_mask_file, xfm_file, inv_xfm_file]:
                img = file.format(participant_id=ppid, session=session)
                if not self.load_img(img, level='warning'):
                    self.ppids_bad.append(ppid)
                    break

        self.ppids_bad = list(set(self.ppids_bad))
        self.ppids = list(set(self.ppids) - set(self.ppids_bad))
        return True

    def _connectivity(self):
        """Validate connectivity matrix input data"""
        seed_file = self.data['masks']['seed']
        seed = self.load_img(seed_file, level='error')
        seed_data = seed.get_data()
        seed_coords_file = self.data['seed_coordinates']
        n_voxels = np.count_nonzero(seed_data)

        # Check if seed mask was loaded
        if seed is None:
            return False

        # Validate seed mask (must be binary, cannot be atlas)
        if not np.array_equal(seed_data, seed_data.astype(bool)):
            logging.error('seed mask %s is not binary' % seed_file)

        # validate seed coordinates file
        try:
            self.seed_coordinates = np.load(seed_coords_file)

        except Exception as exc:
            # TODO: More specific exception
            logging.error('unable to read contents of file: %s'
                          % seed_coords_file)
            return False

        if self.seed_coordinates.shape != (n_voxels, 3):
            shape = str(self.seed_coordinates.shape)
            logging.error('expected shape (%s, 3), not %s for seed '
                          'coordinates' % (n_voxels, shape))
            return False

        # validate connectivity matrices
        template_conn = self.data['connectivity']
        sessions = self.data.get('session', [None])

        for ppid, session in itertools.product(self.ppids, sessions):
            if session:
                name = 'subject-id %s, session %s' % (ppid, session)
            else:
                name = 'subject-id %s' % ppid

            file = template_conn.format(participant_id=ppid)

            try:
                _, ext = os.path.splitext(file)

                if ext == '.npz':
                    headers = npz_headers(file)
                    d = {k: v for k, v, _ in headers}
                    if 'connectivity' not in list(d.keys()):
                        logging.warning('cannot find connectivity.npy in %s'
                                        % file)
                        self.ppids_bad.append(ppid)
                        continue

                    shape = d.get('connectivity', None)

                elif ext == '.npy':
                    shape, _ = npy_header(file)

                else:
                    raise ValueError('unknown extension for file %s, must be '
                                     '.npy or .npz' % file)

                if shape[0] != n_voxels:
                    logging.warning('expected connectivity matrix shape '
                                    '(%s, x), not (%s, x) for %s'
                                    % (n_voxels, shape[0], name))
                    self.ppids_bad.append(ppid)

            except:
                # TODO: more specific exception
                logging.warning('Unable to open %s' % file)
                self.ppids_bad.append(ppid)
                continue

        self.ppids_bad = list(set(self.ppids_bad))
        self.ppids = list(set(self.ppids) - set(self.ppids_bad))
        return True

    def get_size(self) -> None:
        """Estimate size of interim output"""
        if self.modality in ('rsfmri', 'dmri'):
            target_size = self.n_target_voxels
            conn_size = self.n_seed_voxels * target_size * len(self.ppids)
            conn_size = readable_bytesize(conn_size, 8)
            logging.info('Approximate size of all connectivity matrices: %s'
                         % conn_size)

        cluster_size = self.n_seed_voxels * len(self.ppids)
        cluster_size += self.n_seed_voxels * 2
        cluster_size += len(self.ppids) + 1
        cluster_size = readable_bytesize(cluster_size, 8)
        logging.info('Approximate size of all cluster label files: %s'
                     % cluster_size)

    def get_mem(self) -> None:
        """Estimate memory usage per task"""
        mem_mb_conn = 0
        mem_mb_clust = 0
        sessions = self.data.get('session', [None])

        # Connectivity task
        if self.modality == 'rsfmri':
            time_series = self.data['time_series']
            sizes = [
                os.path.getsize(time_series.format(
                    participant_id=ppid, session=session
                ))
                for ppid, session in itertools.product(self.ppids, sessions)
            ]
            mem_mb_conn = bytes_to(np.ceil(max(sizes) * 2.5), 'mb')
            mem_mb_conn = int(np.ceil(mem_mb_conn))
            mem_mb_conn += self.mem_padding

        elif self.modality == 'dmri':
            samples = self.data['samples'] + '*'
            sizes = []

            for ppid, session in itertools.product(self.ppids, sessions):
                sizes.append(sum([
                    os.path.getsize(sample)
                    for sample in glob.glob(
                        samples.format(participant_id=ppid, session=session)
                    )
                ]))
            mem_mb_conn = bytes_to(np.ceil(max(sizes) * 2.5), 'mb')
            mem_mb_conn = int(np.ceil(mem_mb_conn))
            mem_mb_conn += self.mem_padding

        # Clustering task
        if self.modality in ('rsfmri', 'dmri'):
            seed = self.n_seed_voxels
            target = self.n_target_voxels
            mem_mb_clust = bytes_to(seed * target * len(self.ppids), 'mb')
            mem_mb_clust = int(np.ceil(mem_mb_clust))
            mem_mb_clust += self.mem_padding

        elif self.modality == 'connectivity':
            # TODO: If .npz, then estimate larger size!
            connectivity = self.data['connectivity']
            sizes = [
                os.path.getsize(connectivity.format(
                    participant_id=ppid, session=session
                ))
                for ppid, session in itertools.product(self.ppids, sessions)
            ]
            mem_mb_clust = bytes_to(np.ceil(max(sizes)), 'mb')
            mem_mb_clust = int(np.ceil(mem_mb_clust))
            mem_mb_clust += self.mem_padding

            # TODO: temporary padding until above to-do resolved
            mem_mb_clust += 750

        self.mem_mb['connectivity'] = mem_mb_conn
        self.mem_mb['clustering'] = mem_mb_clust

    def _references(self, imgs: list) -> bool:
        """Validate reference images to ensure they are valid nifti images
        that have the same space as the input seed mask.

        References should also at least have 2 clusters.
        """

        # TODO: references can still be different after mask processing
        # basically: median filtering cannot be used if references are given
        #   unless references are also median filtered

        seed = self.data['masks']['seed']
        seed = self.load_img(seed, level=None)
        if seed is None:
            logging.error('cannot validate references because the seed mask '
                          'could not be loaded')
            return False

        valid = True
        for file in imgs:
            # Check if image can be loaded
            img = self.load_img(file, level='error')
            if img is None:
                valid = False
                continue

            # Check if image is in the same space as the seed
            if not imgs_equal_3d([img, seed]):
                valid = False
                logging.error('reference \'%s\' is not in the same space as '
                              'the seed mask' % file)
                continue

            # Check if image covers the same voxels as the seed
            data = img.get_data()
            ref_coords = np.asarray(tuple(zip(*np.where(data > 0))))
            seed_coords = np.asarray(
                tuple(zip(*np.where(seed.get_data() > 0))))

            if not np.array_equal(ref_coords, seed_coords):
                valid = False
                logging.error('reference \'%s\' does not cover the same '
                              'voxels as the seed mask' % file)
                continue

            # Check if image has integer-based cluster-ids
            if not np.all(data == data.astype(int)):
                valid = False
                logging.error('reference \'%s\' does not have integer-based'
                              'cluster-ids (they may be floats)' % file)
                continue

            # Check if the reference image has at least 2 clusters
            # That is, 3 unique values (including 0 for background)
            if np.unique(data).size <= 2:
                valid = False
                logging.error('No clusters found in reference \'%s\'' % file)
                continue

        return valid

    def validate(self) -> bool:
        """Validate input data set"""
        # Retrieve and validate participant-ids
        self.ppids = self.get_ppids(**self.data['participants'])
        if len(self.ppids) == 0:
            return False

        # Validate modality-specific input data
        if hasattr(self, '_%s' % self.modality):
            data_validation = getattr(self, '_%s' % self.modality)
            if not data_validation():
                return False
        else:
            raise ValueError('method self._%s not found' % self.modality)

        # Validate reference images
        self.references = self.data.get('references', None)
        if self.references:
            if not self._references(self.references):
                return False

        if not len(self.ppids) > 0:
            logging.error('No subjects left after removing those with missing '
                          'or bad data')
            return False

        self.get_mem()
        self.get_size()
        return True


class Setup:
    def __init__(self, document: dict):
        """Validates and prepares data and configuration"""
        self.document = document
        self.modality = document.get('modality', None)
        self.data_set = None

    @staticmethod
    def _fsl_available():
        """Check if FSL's probtrackx2 is accessible"""
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
            logging.warning('no module named \'probtrackx2\' (FSL)')

    def overview(self):
        """Provide an overview of errors and warnings during setup"""
        if logging.error.count > 0:
            display('%s error(s) during setup {red}[ERROR]{endc}'
                    % logging.error.count)

        if logging.warning.count > 0:
            display('%s warning(s) during setup {yellow}[WARNING]{endc}'
                    % logging.warning.count)

        if len(self.data_set.ppids_bad) > 0:
            display('%s participant(s) removed due to missing or bad data '
                    '{yellow}[WARNING]{endc}' % len(self.data_set.ppids_bad))

    def save(self, workdir: str) -> None:
        """Save/copy files to project directory"""
        # Create folder structure
        os.makedirs(workdir, exist_ok=True)
        os.makedirs(os.path.join(workdir, 'log'), exist_ok=True)

        # Create lookup file
        if len(self.data_set.ppids) > 0:
            df = pd.DataFrame(self.data_set.ppids, columns=['participant_id'])
            df.sort_values(by='participant_id', inplace=True)
            fpath = os.path.join(workdir, 'participants.tsv')
            df.to_csv(fpath, sep='\t', index=False)

        if len(self.data_set.ppids_bad) > 0:
            df = pd.DataFrame(self.data_set.ppids_bad,
                              columns=['participant_id'])
            df.sort_values(by='participant_id', inplace=True)
            fpath = os.path.join(workdir, 'participants_bad.tsv')
            df.to_csv(fpath, sep='\t', index=False)

        if isinstance(self.data_set.seed_coordinates, np.ndarray):
            fpath = os.path.join(workdir, 'seed_coordinates.npy')
            # np.save(fpath, self.data_set.seed_coordinates)
            np.save(fpath, self.data_set.data['seed_coordinates'])
            logging.info('created file %s' % fpath)

        if isinstance(self.data_set.references, list):
            os.makedirs(os.path.join(workdir, 'references'), exist_ok=True)
            for reference in self.data_set.references:
                fname = os.path.basename(reference)
                fpath = os.path.join(workdir, 'references', fname)
                shutil.copy(reference, fpath)
                logging.info('copied file %s' % fpath)

        logging.info('Removed participants: %s' % len(self.data_set.ppids_bad))
        logging.info('Included participants: %s' % len(self.data_set.ppids))

        # Create the workflow
        document = {
            'modality': self.modality,
            'data': self.data_set.data.copy(),
            'parameters': self.document['parameters'].copy(),
            'mem_mb': {
                'connectivity': self.data_set.mem_mb.get('connectivity'),
                'clustering': self.data_set.mem_mb.get('clustering')
            }
        }
        build_workflow(document, save_at=workdir)

        # Save the final configuration file
        fpath = os.path.join(workdir, 'configuration.yaml')
        with open(fpath, 'w') as f:
            yaml.dump(self.document, f, default_flow_style=False)

    def process(self) -> bool:
        """Start the data validation procedure"""
        logging.info('starting %s data validation' % self.modality)
        if self.modality == 'dmri':
            self._fsl_available()

        self.data_set = DataSet(document=self.document)
        data_is_valid = self.data_set.validate()

        if not data_is_valid:
            return False

        space = self.document.get('masks', {}).get('space')
        if not len(self.data_set.ppids) > 1 and space == 'standard':
            # Standard space does group analysis, thus needing at least 2 pps
            logging.error('not enough participants left to continue '
                          '(the bare minimum is 2)')
            return False
        elif not len(self.data_set.ppids) > 0:
            # Native space can work with just 1 pp
            logging.error('no participants left to process')
            return False

        return True


def display(msg, *args, **kwargs):
    """Use terminal colors when printing"""
    print(msg.format(
        red=TColor.red,
        blue=TColor.blue,
        yellow=TColor.yellow,
        green=TColor.green,
        bold=TColor.bold,
        endc=TColor.reset_all
    ), **kwargs)
