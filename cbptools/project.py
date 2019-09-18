from cbptools.utils import readable_bytesize, CallCountDecorator, TColor, \
    bytes_to, npy_header, npz_headers
from cbptools.image import binarize_3d, stretch_img, median_filter_img, \
    subtract_img, subsample_img, imgs_equal_3d, get_mask_indices
from nibabel.processing import resample_from_to, vox2out_vox
from nibabel.filebasedimages import ImageFileError
from nibabel.spatialimages import SpatialImage
from pandas.io.common import EmptyDataError
from typing import Union
import nibabel as nib
import pandas as pd
import numpy as np
import pkg_resources
import logging
import fnmatch
import shutil
import yaml
import glob
import os


class DataSet:
    def __init__(self, document: dict):
        self.modality = document.get('modality', None)
        self.data = document.get('data', {})
        self.parameters = document.get('parameters', {})
        self.template = pkg_resources.resource_filename(
            __name__, 'templates/MNI152GM.nii.gz')
        self.template = self.template
        self.is_validated = False
        self.target = None
        self.seed = None
        self.highres_seed = None
        self.seed_coordinates = None
        self.mem_mb = {'connectivity': 0, 'clustering': 0}
        self.ppids = []
        self.ppids_bad = []

    @staticmethod
    def _get_subject_ids(file: str, delimiter: str = None,
                         index_column: str = None) -> list:
        if not os.path.exists(file):
            logging.error('No such file: %s' % file)
            return []

        df = pd.read_csv(file, sep=delimiter, engine='python')

        if delimiter is None:
            logging.warning('no delimiter set, inferred delimiter is %s'
                            % df._engine.data.dialect.delimiter)

        if index_column is None:
            index_column = 'participant_id'
            logging.warning('no index_column set, inferred index_column is %s'
                            % index_column)

        if index_column not in df.columns:
            logging.error(
                'index_column %s does not exist in %s' % (index_column, file))
            return []

        ppids = list(df.get(index_column))

        if len(ppids) == 0:
            logging.error('no subject-ids found in %s' % file)

        return list(set(ppids))

    @staticmethod
    def load_img(file: str, level: str = None):
        try:
            return nib.load(file)

        except (ImageFileError, FileNotFoundError) as exc:
            if level == 'error':
                logging.error(exc)
            elif level == 'warning':
                logging.warning(exc)

            return None

    @staticmethod
    def _validate_imgs(seed: SpatialImage, target: SpatialImage) -> bool:
        equal_affines = np.equal(seed.affine, target.affine)
        equal_shapes = np.equal(seed.shape, target.shape)

        if not equal_affines or not equal_shapes:
            logging.error('seed- and target-mask are not in the same space')
            return False

        return True

    def _validate_rsfmri(self) -> bool:
        seed = self.load_img(self.data['seed_mask'], level='error')
        target = self.data.get('target_mask', None)

        if target is None:
            logging.warning('using default 2mm isotropic MNI152 gray matter '
                            'template as target')
            target = self.load_img(self.template, level='error')
        else:
            target = self.load_img(target, level='error')

        if target is None or seed is None:
            return False

        if not imgs_equal_3d([seed, target]):
            logging.error('seed- and target-mask are not in the same space')
            return False

        # validate time-series and confounds
        template_time_series = self.data['time_series']
        template_confounds = self.data.get('confounds', None)

        if isinstance(template_confounds, dict):
            sep = template_confounds.get('delimiter', None)
            usecols = template_confounds.get('columns', None)

            if sep is None:
                sep = '\t'
                logging.warning('no confounds delimiter set, using default '
                                'tab')

            if not usecols:
                logging.warning('no confounds columns selected, including all')

        for ppid in self.ppids:
            time_series = template_time_series.format(participant_id=ppid)
            time_series_img = self.load_img(time_series)

            if time_series_img is None:
                logging.warning(
                    'required file(s) is/are missing for subject-id %s' % ppid)
                self.ppids_bad.append(ppid)
                continue

            if not imgs_equal_3d([time_series_img, seed]):
                logging.warning('time_series and seed_mask are not in the '
                                'same space for subject-id %s' % ppid)
                self.ppids_bad.append(ppid)
                continue

            if isinstance(template_confounds, dict):
                try:
                    df = pd.read_csv(
                        template_confounds['file'].format(participant_id=ppid),
                        sep=sep, engine='python')
                except (EmptyDataError, FileNotFoundError) as exc:
                    logging.warning(exc)
                    self.ppids_bad.append(ppid)
                    continue

                if usecols:
                    header = df.columns.tolist()
                    xusecols = []
                    for col in usecols:
                        if '*' in col:
                            xusecols += [x for x in header if any(
                                fnmatch.fnmatch(x, p) for p in [col])]
                        else:
                            xusecols.append(col)

                    usecols = [i for i in xusecols if i is not None]

                    if not set(usecols).issubset(set(header)):
                        missing = list(set(usecols) - set(header))
                        logging.warning(
                            'missing confounds columns for subject-id %s: %s'
                            % (ppid, ', '.join(missing)))
                        self.ppids_bad.append(ppid)
                        continue

                if len(df) != time_series_img.shape[-1]:
                    logging.warning('confounds and time-series for subject-id '
                                    '%s do not have matching timepoints'
                                    % ppid)
                    self.ppids_bad.append(ppid)
                    continue

        self.seed = seed
        self.target = target
        self.ppids_bad = list(set(self.ppids_bad))
        self.ppids = list(set(self.ppids) - set(self.ppids_bad))
        return True

    def _validate_dmri(self) -> bool:
        seed = self.load_img(self.data['seed_mask'], level='error')
        target = self.data.get('target_mask', None)

        if target is None:
            logging.warning('using default 2mm isotropic MNI152 gray matter '
                            'template as target')
            target = self.load_img(self.template, level='error')
        else:
            target = self.load_img(target, level='error')

        if seed is None or target is None:
            return False

        if not imgs_equal_3d([seed, target]):
            logging.error('seed- and target-mask are not in the same space')
            return False

        # validate bedpostx output
        bet_binary_mask = self.data['bet_binary_mask']
        xfm = self.data['xfm']
        inv_xfm = self.data['inv_xfm']
        samples = self.data['samples']

        # TODO: validate more thoroughly

        for ppid in self.ppids:
            merged = glob.glob(samples.format(participant_id=ppid) + '*')
            if not merged:
                logging.warning('file(s) are missing for subject-id %s' % ppid)
                self.ppids_bad.append(ppid)

            else:
                for file in [bet_binary_mask, xfm, inv_xfm]:
                    if self.load_img(file.format(participant_id=ppid)) is None:
                        logging.warning(
                            'file(s) are missing for subject-id %s' % ppid)
                        self.ppids_bad.append(ppid)
                        break

        self.seed = seed
        self.target = target
        self.ppids_bad = list(set(self.ppids_bad))
        self.ppids = list(set(self.ppids) - set(self.ppids_bad))
        return True

    def _validate_connectivity(self):
        seed = self.load_img(self.data['seed_mask'], level='error')

        if seed is None:
            return False

        n_voxels = np.count_nonzero(seed.get_data())

        # validate seed coordinates file
        seed_coords_file = self.data['seed_coordinates']

        try:
            self.seed_coordinates = np.load(seed_coords_file)

        except Exception as exc:
            logging.error('unable to read contents of file: %s'
                          % seed_coords_file)
            return False

        if self.seed_coordinates.shape != (n_voxels, 3):
            logging.error('expected shape (%s, 3), not %s for seed '
                          'coordinates' % (n_voxels,
                                           str(self.seed_coordinates.shape)))
            return False

        # validate connectivity matrices
        template_conn = self.data['connectivity']

        for ppid in self.ppids:
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
                                    '(%s, x), not (%s, x) for subject-id %s'
                                    % (n_voxels, shape[0], ppid))
                    self.ppids_bad.append(ppid)

            except:  # TODO: make exception more specific
                logging.warning('Unable to open %s' % file)
                self.ppids_bad.append(ppid)
                continue

        self.seed = seed
        self.ppids_bad = list(set(self.ppids_bad))
        self.ppids = list(set(self.ppids) - set(self.ppids_bad))
        return True

    @staticmethod
    def _preproc_binarize(field: str, img: SpatialImage,
                          bin_threshold: float = 0.0) -> SpatialImage:
        data = img.get_data()

        if not np.array_equal(data, data.astype(bool)):
            n_nans = np.count_nonzero(np.isnan(data))
            n_infs = np.count_nonzero(np.isinf(data))

            if n_nans > 0:
                logging.warning(
                    '%s NaN values in %s. Please manually verify %s.nii'
                    % (n_nans, field, field))

            if n_infs > 0:
                logging.warning(
                    '%s inf values in %s. Please manually verify %s.nii'
                    % (n_infs, field, field))

        img = binarize_3d(img, threshold=bin_threshold)
        return img

    def _preprocess_rsfmri(self, parameters: dict) -> None:
        mask_preproc_seed = parameters.get('mask_preproc_seed', {})
        mask_preproc_target = parameters.get('mask_preproc_target', {})
        subsampling = mask_preproc_target['subsampling']

        if subsampling:
            logging.info('applying subsampling on target_mask')
            self.target = subsample_img(self.target, f=2)

    def _preprocess_dmri(self, parameters: dict) -> None:
        mask_preproc_seed = parameters.get('mask_preproc_seed', {})
        mask_preproc_target = parameters.get('mask_preproc_target', {})

        upsampling = mask_preproc_seed['upsample_to']['apply']
        vox_dim = mask_preproc_seed['upsample_to']['voxel_dimensions']

        if upsampling:
            if len(vox_dim) == 1:
                vox_dim *= 3

            logging.info('stretching seed_mask to %s'
                         % 'x'.join(map(str, vox_dim)))
            mapping = list(
                vox2out_vox((self.seed.shape, self.seed.affine), vox_dim))
            a = np.sign(self.seed.affine)
            b = np.sign(mapping[1])
            mapping[1] *= (a * b)
            self.highres_seed = stretch_img(self.seed, mapping)

        # Target
        downsampling = mask_preproc_target['downsample_to']['apply']
        vox_dim = mask_preproc_target['downsample_to']['voxel_dimensions']

        if downsampling:
            if len(vox_dim) == 1:
                vox_dim *= 3

            logging.info('resampling target_mask to %s'
                         % 'x'.join(map(str, vox_dim)))
            mapping = list(vox2out_vox((
                self.target.shape, self.target.affine), vox_dim))
            a = np.sign(self.target.affine)
            b = np.sign(mapping[1])
            mapping[1] *= (a * b)
            self.target = resample_from_to(self.target, mapping, order=0,
                                           mode='nearest')

    def _memory_estimate(self):
        buffer = 250  # in MB
        mem_mb_conn = 0
        mem_mb_clust = 0

        # Connectivity task
        if self.modality == 'rsfmri':
            time_series = self.data['time_series']
            sizes = [
                os.path.getsize(time_series.format(participant_id=ppid))
                for ppid in self.ppids
            ]
            mem_mb_conn = bytes_to(np.ceil(max(sizes) * 2.5), 'mb')
            mem_mb_conn = int(np.ceil(mem_mb_conn))
            mem_mb_conn += buffer

        elif self.modality == 'dmri':
            samples = self.data['samples'] + '*'
            sizes = []
            for ppid in self.ppids:
                sizes.append(sum([
                    os.path.getsize(sample)
                    for sample in glob.glob(
                        samples.format(participant_id=ppid)
                    )
                ]))
            mem_mb_conn = bytes_to(np.ceil(max(sizes) * 2.5), 'mb')
            mem_mb_conn = int(np.ceil(mem_mb_conn))
            mem_mb_conn += buffer

        # Clustering task
        if self.modality in ('rsfmri', 'dmri'):
            seed = np.count_nonzero(self.seed.get_data())
            target = np.count_nonzero(self.target.get_data())
            mem_mb_clust = bytes_to(seed * target * len(self.ppids), 'mb')
            mem_mb_clust = int(np.ceil(mem_mb_clust))
            mem_mb_clust += buffer

        elif self.modality == 'connectivity':
            # TODO: If .npz, then estimate larger size!
            connectivity = self.data['connectivity']
            sizes = [
                os.path.getsize(connectivity.format(participant_id=ppid))
                for ppid in self.ppids
            ]
            mem_mb_clust = bytes_to(np.ceil(max(sizes)), 'mb')
            mem_mb_clust = int(np.ceil(mem_mb_clust))
            mem_mb_clust += buffer

            # TODO: temporary until above to-do resolved
            mem_mb_clust += 750

        self.mem_mb['connectivity'] = mem_mb_conn
        self.mem_mb['clustering'] = mem_mb_clust

    def validate(self) -> bool:
        self.ppids = self._get_subject_ids(**self.data['participants'])
        if len(self.ppids) == 0:
            return False

        if hasattr(self, '_validate_%s' % self.modality):
            data_validation = getattr(self, '_validate_%s' % self.modality)
            if not data_validation():
                return False
        else:
            raise ValueError(
                'method self._validate_%s not found' % self.modality)

        if not len(self.ppids) > 0:
            logging.error('No subjects left after removing those with missing '
                          'or bad data')
            return False

        self.is_validated = True
        return True

    def preprocess(self) -> None:
        if not self.is_validated:
            raise ValueError('cannot initiate preprocessing before validation')

        if self.modality == 'connectivity':
            self._memory_estimate()
            return

        mask_preproc_seed = self.parameters.get('mask_preproc_seed', {})
        mask_preproc_target = self.parameters.get('mask_preproc_target', {})

        # Seed
        bin_thresh = mask_preproc_seed['binarization']
        medfilt = mask_preproc_seed['median_filtering']['apply']
        medfilt_dist = mask_preproc_seed['median_filtering']['distance']
        self.seed = self._preproc_binarize('seed_mask', self.seed, bin_thresh)

        if medfilt:
            logging.info('applying median filter on seed_mask (distance=%s)'
                         % medfilt_dist)
            self.seed = median_filter_img(self.seed, dist=medfilt_dist)

        # Target
        bin_thresh = mask_preproc_target['binarization']
        remove_seed = mask_preproc_target['remove_seed']['apply']
        remove_seed_dist = mask_preproc_target['remove_seed']['distance']

        self.target = self._preproc_binarize('target_mask', self.target,
                                             bin_thresh)

        if remove_seed:
            logging.info('removing seed_mask from target_mask (distance=%s)'
                         % remove_seed_dist)
            self.target = subtract_img(self.target, self.seed,
                                       remove_seed_dist)

        if hasattr(self, '_preprocess_%s' % self.modality):
            preproc = getattr(self, '_preprocess_%s' % self.modality)
            preproc(self.parameters)
        else:
            raise ValueError(
                'method self._preprocess_%s not found' % self.modality)

        # Get seed voxel coordinates
        self.seed_coordinates = get_mask_indices(self.seed, order='C')
        self._memory_estimate()


class Workflow:
    def __init__(self, workdir, modality):
        self.workdir = workdir
        self.modality = modality
        self.templates = pkg_resources.resource_filename(__name__, 'templates')
        self.workflow = os.path.join(self.workdir, 'Snakefile')
        self.cluster_json = os.path.join(self.templates, 'cluster.json')
        self.snakefiles = ['header.Snakefile', 'body.Snakefile']

    def _get(self, data: dict, *args: str) -> Union[str, None]:
        if not isinstance(data, dict):
            return None

        return data.get(args[0], None) if len(args) == 1 \
            else self._get(data.get(args[0], {}), *args[1:])

    def _parse(self, line: str, document: dict) -> Union[str, bool]:
        """Parse <cbptools['key1:key2:key3']> string"""

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
            value = self._get(document, *keys)

            if isinstance(value, dict):
                value = value if not value.get('file', None) else value.get(
                    'file')

            # TODO: Check if False can be returned for any value that is None
            if keys[0] == 'data' and not value:
                return False

            if inplace:
                value = repr(value).encode('utf-8').decode('unicode_escape') \
                    if isinstance(value, str) else str(value)
                line = line.replace('%s!%s%s' % (s, content, e), value)

            elif force:
                line = line.replace('%s+%s%s' % (s, content, e), str(value))

            else:
                value = repr(value) if isinstance(value, str) else str(value)
                line = line.replace(
                    '%s%s%s' % (s, content, e),
                    '%s = %s' % (keys[-1], value)
                )

        return line

    def create(self, data: dict, parameters: dict, mem_mb: tuple) -> None:
        # Add temporary fields for creaeting the workflow
        if self.modality in ('rsfmri', 'dmri'):
            self.snakefiles.insert(1, '%s.Snakefile' % self.modality)
            data['connectivity'] = \
                'connectivity/connectivity_{participant_id}.npz'
            data['seed_coordinates'] = 'seed_coordinates.npy'

        if self.modality == 'rsfmri':
            data['touchfile'] = 'log/.touchfile'

        if self.modality == 'dmri':
            pd = parameters.get(
                'probtract_proc', {}).get('correct_path_distribution', False)
            loop_check = parameters.get(
                'probtract_proc', {}).get('loop_check', False)
            parameters['probtract_proc']['correct_path_distribution'] = \
                '--pd' if pd else ''
            parameters['probtract_proc']['loop_check'] = \
                '-l' if loop_check else ''

        document = {
            'data': data,
            'parameters': parameters,
            'mem_mb': {'connectivity': mem_mb[0], 'clustering': mem_mb[1]}
        }

        # Parse workflow
        with open(self.workflow, 'w') as sf:
            for snakefile in self.snakefiles:
                template = os.path.join(self.templates, snakefile)
                with open(template, 'r') as f:
                    for line in f:
                        line = self._parse(line, document)

                        if line is not False:
                            sf.write(line)

        # Copy cluster.json file
        shutil.copy(self.cluster_json, self.workdir)


class Setup:
    """Validates and prepares data and configuration"""

    def __init__(self, document: dict):
        logging.error = CallCountDecorator(logging.error)
        logging.warning = CallCountDecorator(logging.warning)
        self.document = document
        self.modality = document.get('modality', None)
        self.data_set = None

    @staticmethod
    def _fsl_available():
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
        if logging.error.count > 0:
            display('%s error(s) during setup' % logging.error.count)

        if logging.warning.count > 0:
            display('%s warning(s) during setup' % logging.warning.count)

        if len(self.data_set.ppids_bad) > 0:
            display('%s participant(s) removed due to missing or bad data'
                    % len(self.data_set.ppids_bad))

    def save(self, workdir: str) -> None:
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

        if isinstance(self.data_set.seed, SpatialImage):
            fpath = os.path.join(workdir, 'seed_mask.nii.gz')
            nib.save(self.data_set.seed, fpath)
            logging.info('created file %s' % fpath)

        if isinstance(self.data_set.target, SpatialImage):
            fpath = os.path.join(workdir, 'target_mask.nii.gz')
            nib.save(self.data_set.target, fpath)
            logging.info('created file %s' % fpath)

        if isinstance(self.data_set.highres_seed, SpatialImage):
            fpath = os.path.join(workdir, 'highres_seed_mask.nii.gz')
            nib.save(self.data_set.highres_seed, fpath)
            logging.info('created file %s' % fpath)

        if isinstance(self.data_set.seed_coordinates, np.ndarray):
            fpath = os.path.join(workdir, 'seed_coordinates.npy')
            np.save(fpath, self.data_set.seed_coordinates)
            logging.info('created file %s' % fpath)

        logging.info('Removed participants: %s' % len(self.data_set.ppids_bad))
        logging.info('Included participants: %s' % len(self.data_set.ppids))

        # Estimate size of interim output
        seed_size = np.count_nonzero(self.data_set.seed.get_data())

        if self.modality in ('rsfmri', 'dmri'):
            target_size = np.count_nonzero(self.data_set.target.get_data())
            conn_size = seed_size * target_size * len(self.data_set.ppids)
            conn_size = readable_bytesize(conn_size, 8)
            logging.info('Approximate size of all connectivity matrices: %s'
                         % conn_size)

        cluster_size = seed_size * len(self.data_set.ppids)
        cluster_size += seed_size * 2
        cluster_size += len(self.data_set.ppids) + 1
        cluster_size = readable_bytesize(cluster_size, 8)
        logging.info(
            'Approximate size of all cluster label files: %s' % cluster_size)

        # Create the workflow
        mem_mb = (self.data_set.mem_mb.get('connectivity'),
                  self.data_set.mem_mb.get('clustering'))
        workflow = Workflow(workdir, self.modality)
        workflow.create(self.data_set.data.copy(),
                        self.data_set.parameters.copy(), mem_mb)

        # Save the final configuration file
        fpath = os.path.join(workdir, 'configuration.yaml')
        with open(fpath, 'w') as f:
            yaml.dump(self.document, f, default_flow_style=False)

    def process(self) -> bool:
        logging.info('Selected modality for input validation: %s'
                     % self.modality)
        if self.modality == 'dmri':
            self._fsl_available()

        self.data_set = DataSet(document=self.document)
        data_is_valid = self.data_set.validate()

        if not data_is_valid:
            return False

        if not len(self.data_set.ppids) >= 2:
            logging.error('not enough participants left to continue '
                          '(the bare minimum is 2)')
            return False

        self.data_set.preprocess()
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
