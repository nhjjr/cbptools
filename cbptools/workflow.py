from .utils import config_get
from os.path import join as opj, basename as opb


class BaseRule(object):
    jid = '%j'
    dependencies = None
    name = None

    def __init__(self, doc):
        self._doc = doc

        compress = self.get('parameters.report.compress_output', False)
        self.nifti_ext = 'nii.gz' if compress else 'nii'

    def is_active(self):
        return True

    def get(self, keymap, default=None):
        return config_get(keymap, self._doc, default)

    @staticmethod
    def fwrap(parts, f: str):
        if isinstance(parts, str) or isinstance(parts, int):
            return 'f:%s(\'%s\')' % (f, parts)

        elif not isinstance(parts, list):
            parts = list(parts)

        if len(parts) <= 1:
            return 'f:%s(\'%s\')' % (f, parts[0])

        else:
            line = 'f:%s(\'%s\'' % (f, parts[0])

            for part in parts[1:]:
                line += ', %s' % part

            line += ')'
            return line

    @staticmethod
    def wildcard(value) -> str:
        return 'f:lambda wildcards: %s' % value

    @property
    def input(self):
        return None

    @property
    def output(self):
        return None

    @property
    def log(self):
        return None

    @property
    def benchmark(self):
        return None

    @property
    def cluster_json(self):
        return None

    @property
    def threads(self):
        return None

    @property
    def resources(self):
        return None

    @property
    def params(self):
        return None

    @property
    def run(self):
        return None

    @property
    def shell(self):
        return None


class RuleAll(BaseRule):
    name = 'all'

    def __init__(self, conf):
        super().__init__(conf)

    @property
    def input(self):
        d = list()

        # Parameter keys & files
        space = 'data.masks.space'
        figformat = 'parameters.report.figure_format'
        b_ind_plots = 'parameters.report.individual_plots'
        n_clusters = 'parameters.clustering.n_clusters'
        intval_metrics = 'parameters.clustering.validity.internal'
        references = 'data.references'
        space = self.get(space, 'standard')
        intval_metrics = self.get(intval_metrics, None)
        figformat = self.get(figformat)
        b_ind_plots = self.get(b_ind_plots, False)
        n_clusters = self.get(n_clusters)
        n_clusters = 'n_clusters=%s' % n_clusters
        ppid = 'participant_id=participants'
        views = ['right', 'left', 'superior', 'inferior', 'posterior',
                 'anterior']
        views = 'view=%s' % views

        path = 'group'
        gsim_tsv = opj(path, 'group_similarity.tsv')
        coph_tsv = opj(path, 'cophenetic_correlation.tsv')
        gsim_plot = opj(path, 'group_similarity.%s' % figformat)
        accuracy = opj(path, 'relabeling_accuracy.%s' % figformat)
        coph_plot = opj(path, 'cophenetic_correlation.%s' % figformat)
        refsim_tsv = opj(path, 'reference_similarity.tsv')
        refsim_plot = opj(path, 'reference_similarity.%s' % figformat)

        path = opj(path, '{n_clusters}clusters')
        labels = opj(path, 'labels.npz')
        labeled_roi = opj(path, 'labeled_roi.%s' % self.nifti_ext)
        heatmap = opj(path, 'individual_similarity_heatmap.png')
        clustermap = opj(path, 'individual_similarity_clustermap.png')
        ind_sim = opj(path, 'individual_similarity.npy')
        group_voxel_plot = opj(path, 'voxel_plot')

        path = 'individual'
        intval_tsv = opj(path, 'internal_validity.tsv')
        intval_plot = opj(path, 'internal_validity')

        path = opj(path, '{participant_id}')
        ind_voxel_plot = opj(path, '{n_clusters}cluster_voxel_plot')

        # Define parameters
        if space == 'standard':
            d.append(gsim_tsv)
            d.append(coph_tsv)
            d.append(gsim_plot)
            d.append(accuracy)
            d.append(coph_plot)
            d.append(self.fwrap([labels, n_clusters], 'expand'))
            d.append(self.fwrap([labeled_roi, n_clusters], 'expand'))
            d.append(self.fwrap([heatmap, n_clusters], 'expand'))
            d.append(self.fwrap([clustermap, n_clusters], 'expand'))
            d.append(self.fwrap([ind_sim, n_clusters], 'expand'))

            if self.get(references, None):
                d.append(refsim_tsv)
                d.append(refsim_plot)

            plot = '%s_{view}.%s' % (group_voxel_plot, figformat)
            d.append(self.fwrap([plot, n_clusters, views], 'expand'))

        if space == 'native' or b_ind_plots:
            plot = '%s_{view}.%s' % (ind_voxel_plot, figformat)
            d.append(self.fwrap([plot, ppid, n_clusters, views], 'expand'))

        if intval_metrics:
            d.append(intval_tsv)
            intval_metrics = 'metric=%s' % intval_metrics
            plot = '%s_{metric}.%s' % (intval_plot, figformat)
            d.append(self.fwrap([plot, intval_metrics], 'expand'))

        return d


class RuleProcessMasksRSFMRI(BaseRule):
    name = 'process_masks_rsfmri'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        modality = self.get('modality')
        return True if modality == 'rsfmri' else False

    @property
    def input(self):
        d = dict()

        # Parameter keys
        seed_mask = self.get('data.masks.seed')
        target_mask = self.get('data.masks.target', None)

        d['seed_mask'] = seed_mask
        if target_mask:
            d['target_mask'] = target_mask

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        space = self.get('data.masks.space', 'standard')
        seed_img = 'seed_mask.%s' % self.nifti_ext
        target_img = 'target_mask.%s' % self.nifti_ext
        seed_coords = 'seed_coordinates.npy'

        # Define output
        if space == 'native':
            path = 'individual/{participant_id}'
            d['seed_img'] = opj(path, seed_img)
            d['target_img'] = opj(path, target_img)
            d['seed_coordinates'] = opj(path, seed_coords)
        else:  # if space == 'standard'
            d['seed_img'] = seed_img
            d['target_img'] = target_img
            d['seed_coordinates'] = seed_coords

        return d

    @property
    def threads(self):
        return 1

    @property
    def log(self):
        return 's:log/%s.log' % self.name

    @property
    def benchmark(self):
        return 's:benchmarks/%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        d['out'] = 'log/{rule}-%s.out' % self.jid
        return d

    @property
    def params(self):
        d = dict()

        # Parameter keys
        region_id = 'data.masks.region_id'
        bin_seed = 'parameters.masking.seed.binarization'
        b_medfilt = 'parameters.masking.seed.median_filtering.apply'
        medfilt = 'parameters.masking.seed.median_filtering.distance'
        bin_target = 'parameters.masking.target.binarization'
        b_remseed = 'parameters.masking.target.remove_seed.apply'
        remseed = 'parameters.masking.target.remove_seed.distance'
        subsample = 'parameters.masking.target.subsampling'

        # Define parameters
        d['bin_seed'] = self.get(bin_seed)
        d['bin_target'] = self.get(bin_target)
        d['subsample'] = self.get(subsample, False)
        region_id = self.get(region_id, None)

        if region_id:
            d['region_id'] = region_id

        if self.get(b_medfilt, False):
            d['median_filter'] = self.get(medfilt)

        if self.get(b_remseed, False):
            d['remove_seed'] = self.get(remseed)

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params, log)' % self.name


class RuleProcessMasksDMRI(BaseRule):
    name = 'process_masks_dmri'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        modality = self.get('modality')
        return True if modality == 'dmri' else False

    @property
    def input(self):
        d = dict()

        # Parameter keys
        seed_mask = self.get('data.masks.seed')
        target_mask = self.get('data.masks.target', None)

        d['seed_mask'] = seed_mask
        if target_mask:
            d['target_mask'] = target_mask

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        upsample = self.get('parameters.masking.seed.upsample_to.apply', False)
        space = self.get('data.masks.space', 'standard')
        seed_img = 'seed_mask.%s' % self.nifti_ext
        highres_seed_img = 'highres_seed_mask.%s' % self.nifti_ext
        target_img = 'target_mask.%s' % self.nifti_ext
        seed_coords = 'seed_coordinates.npy'

        # Define output
        if space == 'native':
            path = 'individual/{participant_id}'
            d['seed_img'] = opj(path, seed_img)
            d['target_img'] = opj(path, target_img)
            d['seed_coordinates'] = opj(path, seed_coords)

            if upsample:
                d['highres_seed_img'] = opj(path, highres_seed_img)

        else:  # if space == 'standard'
            d['seed_img'] = seed_img
            d['target_img'] = target_img
            d['seed_coordinates'] = seed_coords

            if upsample:
                d['highres_seed_img'] = highres_seed_img

        return d

    @property
    def log(self):
        return 's:log/%s.log' % self.name

    @property
    def benchmark(self):
        return 's:benchmarks/%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        d['out'] = 'log/{rule}-%s.out' % self.jid
        return d

    @property
    def threads(self):
        return 1

    @property
    def params(self):
        d = dict()

        # Parameter keys
        region_id = 'data.masks.region_id'
        bin_seed = 'parameters.masking.seed.binarization'
        b_medfilt = 'parameters.masking.seed.median_filtering.apply'
        medfilt = 'parameters.masking.seed.median_filtering.distance'
        bin_target = 'parameters.masking.target.binarization'
        b_remseed = 'parameters.masking.target.remove_seed.apply'
        remseed = 'parameters.masking.target.remove_seed.distance'
        b_upsample = 'parameters.masking.seed.upsample_to.apply'
        upsample = 'parameters.masking.seed.upsample_to.voxel_dimensions'
        b_downsample = 'parameters.masking.target.downsample_to.apply'
        downsample = 'parameters.masking.target.downsample_to.voxel_dimensions'

        # Define parameters
        d['bin_seed'] = self.get(bin_seed)
        d['bin_target'] = self.get(bin_target)

        region_id = self.get(region_id, None)
        if region_id:
            d['region_id'] = region_id

        if self.get(b_medfilt, False):
            d['medfilt'] = self.get(medfilt)

        if self.get(b_upsample, False):
            d['upsample_to'] = self.get(upsample)

        if self.get(b_downsample, False):
            d['downsample_to'] = self.get(downsample)

        if self.get(b_remseed, False):
            d['remove_seed'] = self.get(remseed)

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params, log)' % self.name


class RuleProbtrackx2(BaseRule):
    name = 'probtrackx2'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        modality = self.get('modality')
        return True if modality == 'dmri' else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        upsample = self.get('parameters.masking.seed.upsample_to.apply', False)
        space = self.get('data.masks.space', 'standard')
        bet_binary_mask = self.get('data.bet_binary_mask')
        xfm = self.get('data.xfm', None)
        inv_xfm = self.get('data.inv_xfm', None)
        target_mask = 'target_mask.%s' % self.nifti_ext
        seed_mask = 'highres_seed_mask' if upsample else 'seed_mask'
        seed_mask = '%s.%s' % (seed_mask, self.nifti_ext)

        # Define parameters
        d['bet_binary_mask'] = bet_binary_mask

        if xfm is not None and inv_xfm is not None:
            # note: if one is set, the other must be set too
            d['xfm'] = xfm
            d['inv_xfm'] = inv_xfm

        if space == 'native':
            path = 'individual/{participant_id}'
            d['seed'] = opj(path, seed_mask)
            d['target'] = opj(path, target_mask)
        else:  # if space == 'standard'
            d['seed'] = seed_mask
            d['target'] = target_mask

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        session = self.get('data.session')
        path = 'individual/{participant_id}/probtrackx2'
        path += '_{session}' if session else ''

        # Define parameters
        d['fdt_matrix2'] = opj(path, 'fdt_matrix2.dot')

        return d

    @property
    def benchmark(self):
        session = self.get('data.session', None)
        prfx = '{participant_id}.{session}' if session else '{participant_id}'
        return 's:benchmarks/%s.%s.log' % (prfx, self.name)

    @property
    def cluster_json(self):
        d = dict()
        wildcards = '{wildcards.participant_id}'
        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def resources(self):
        d = dict()

        # Parameter keys
        mem_mb = self.get('mem_mb.connectivity', 1000)
        io = 1

        # Define parameters
        d['mem_mb'] = mem_mb
        d['io'] = io

        return d

    @property
    def params(self):
        d = dict()

        # Parameter keys
        session = self.get('data.session', None)
        samples = 'data.samples'
        dist_thresh = 'parameters.connectivity.dist_thresh'
        c_thresh = 'parameters.connectivity.c_thresh'
        step_length = 'parameters.connectivity.step_length'
        n_samples = 'parameters.connectivity.n_samples'
        n_steps = 'parameters.connectivity.n_steps'
        loop_check = 'parameters.connectivity.loop_check'
        pd = 'parameters.connectivity.correct_path_distribution'
        outdir = 'individual/{participant_id}/probtrackx2'
        outdir += '_{session}' if session else ''

        # Define parameters
        d['outdir'] = outdir
        d['samples'] = self.get(samples)
        d['dist_thresh'] = self.get(dist_thresh)
        d['c_thresh'] = self.get(c_thresh)
        d['step_length'] = self.get(step_length)
        d['n_samples'] = self.get(n_samples)
        d['n_steps'] = self.get(n_steps)

        if self.get(loop_check, False):
            d['loop_check'] = '-l'

        if self.get(pd, False):
            d['correct_path_distribution'] = '--pd'

        return d

    @property
    def shell(self):
        xfm = self.get('data.xfm', None)
        inv_xfm = self.get('data.inv_xfm', None)

        if xfm is None and inv_xfm is None:
            return repr(
                "probtrackx2 --seed={input.seed} --target2={input.target} "
                "--samples={params.samples} --mask={input.bet_binary_mask} "
                "--dir={params.outdir} --nsamples={params.n_samples} "
                "--nsteps={params.n_steps} --steplength={params.step_length} "
                "--distthresh={params.dist_thresh} --cthr={params.c_thresh} "
                "--omatrix2 --forcedir {params.loop_check} "
                "{params.correct_path_distribution} > /dev/null"
            )
        else:
            return repr(
                "probtrackx2 --seed={input.seed} --target2={input.target} "
                "--samples={params.samples} --mask={input.bet_binary_mask} "
                "--xfm={input.xfm} --invxfm={input.inv_xfm} "
                "--dir={params.outdir} --nsamples={params.n_samples} "
                "--nsteps={params.n_steps} --steplength={params.step_length} "
                "--distthresh={params.dist_thresh} --cthr={params.c_thresh} "
                "--omatrix2 --forcedir {params.loop_check} "
                "{params.correct_path_distribution} > /dev/null"
            )


class RuleConnectivityDMRI(BaseRule):
    name = 'connectivity_dmri'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        modality = self.get('modality')
        return True if modality == 'dmri' else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        space = self.get('data.masks.space', 'standard')
        session = self.get('data.session', None)
        path = 'individual/{participant_id}'
        ptx_path = 'probtrackx2'
        ptx_path += '_{session}' if session else ''
        fdt_matrix2 = 'fdt_matrix2.dot'
        upsample = self.get('parameters.masking.seed.upsample_to.apply', False)
        seed_mask = 'highres_seed_mask' if upsample else 'seed_mask'
        seed_mask = '%s.%s' % (seed_mask, self.nifti_ext)

        # Define parameters
        d['fdt_matrix2'] = opj(path, ptx_path, fdt_matrix2)

        if space == 'native':
            d['seed_mask'] = opj(path, seed_mask)
        else:  # if space == 'standard'
            d['seed_mask'] = seed_mask

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        session = self.get('data.session', None)
        path = 'individual/{participant_id}'

        if session:
            connectivity = opj(path, 'connectivity_{session}.npz')
            connectivity = self.fwrap(connectivity, 'temp')
        else:
            connectivity = opj(path, 'connectivity.npz')

        # Define parameters
        d['connectivity'] = connectivity

        return d

    @property
    def log(self):
        session = self.get('data.session', None)
        wildcards = '{participant_id}'
        wildcards += '.{session}' if session else ''
        return 's:log/%s.%s.log' % (wildcards, self.name)

    @property
    def benchmark(self):
        session = self.get('data.session', None)
        wildcards = '{participant_id}'
        wildcards += '.{session}' if session else ''
        return 's:benchmarks/%s.%s.log' % (wildcards, self.name)

    @property
    def cluster_json(self):
        d = dict()

        session = self.get('data.session', None)
        if session:
            wildcards = '{wildcards.participant_id}.{wildcards.session}'
        else:
            wildcards = '{wildcards.participant_id}'

        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def resources(self):
        d = dict()

        # Parameter keys
        mem_mb = self.get('mem_mb.clustering', 1000)

        # Define parameters
        d['mem_mb'] = mem_mb

        return d

    @property
    def params(self):
        d = dict()

        # Parameter keys
        session = self.get('data.session', None)
        cubic = 'parameters.connectivity.cubic_transform.apply'
        b_pca = 'parameters.connectivity.pca_transform.apply'
        pca = 'parameters.connectivity.pca_transform.components'
        compress = 'parameters.report.compress_output'
        cleanup_fsl = 'parameters.connectivity.cleanup_fsl'

        # Define parameters
        d['compress'] = self.get(compress, False)
        d['cleanup_fsl'] = self.get(cleanup_fsl, False)
        d['cubic_transform'] = self.get(cubic, False)

        if not session:
            # For multisession data, PCA transform is applied in merge_sessions
            if self.get(b_pca, False):
                d['pca_transform'] = self.get(pca)
            else:
                d['pca_transform'] = False

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params, log)' % self.name


class RuleConnectivityRSFMRI(BaseRule):
    name = 'connectivity_rsfmri'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        modality = self.get('modality')
        return True if modality == 'rsfmri' else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        space = self.get('data.masks.space', 'standard')
        seed_mask = 'seed_mask.%s' % self.nifti_ext
        target_mask = 'target_mask.%s' % self.nifti_ext
        time_series = self.get('data.time_series')
        confounds_file = self.get('data.confounds.file', None)

        # Define parameters
        d['time_series'] = time_series

        if space == 'native':
            path = 'individual/{participant_id}'
            d['seed_mask'] = opj(path, seed_mask)
            d['target_mask'] = opj(path, target_mask)
        else:  # if space == 'standard'
            d['seed_mask'] = seed_mask
            d['target_mask'] = target_mask

        if confounds_file:
            d['confounds'] = confounds_file

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        session = self.get('data.session', None)
        path = 'individual/{participant_id}'

        # Define parameters
        if session:
            connectivity = 'connectivity_{session}.npz'
            connectivity = opj(path, connectivity)
            d['connectivity'] = self.fwrap(connectivity, 'temp')
        else:
            connectivity = 'connectivity.npz'
            d['connectivity'] = opj(path, connectivity)

        return d

    @property
    def log(self):
        session = self.get('data.session', None)
        wildcards = '{participant_id}'
        wildcards += '.{session}' if session else ''
        return 's:log/%s.%s.log' % (wildcards, self.name)

    @property
    def benchmark(self):
        session = self.get('data.session', None)
        wildcards = '{participant_id}'
        wildcards += '.{session}' if session else ''
        return 's:benchmarks/%s.%s.log' % (wildcards, self.name)

    @property
    def cluster_json(self):
        d = dict()

        session = self.get('data.session', None)
        if session:
            wildcards = '{wildcards.participant_id}.{wildcards.session}'
        else:
            wildcards = '{wildcards.participant_id}'

        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def resources(self):
        d = dict()

        # Parameter keys
        mem_mb = self.get('mem_mb.connectivity', 5000)
        io = 1

        # Define parameters
        d['mem_mb'] = mem_mb
        d['io'] = io

        return d

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        session = self.get('data.session', None)
        participant_id = self.wildcard('str(wildcards.participant_id)')
        arctanh = 'parameters.connectivity.arctanh_transform.apply'
        b_lve = 'parameters.connectivity.low_variance_error.apply'
        lve_seed = 'parameters.connectivity.low_variance_error.in_seed'
        lve_target = 'parameters.connectivity.low_variance_error.in_target'
        compress = 'parameters.report.compress_output'
        confounds_file = 'data.confounds.file'
        confounds_sep = 'data.confounds.delimiter'
        confounds_cols = 'data.confounds.columns'
        b_smoothing = 'parameters.connectivity.smoothing.apply'
        smoothing = 'parameters.connectivity.smoothing.fwhm'
        b_bandpass = 'parameters.connectivity.band_pass_filtering.apply'
        bandpass_band = 'parameters.connectivity.band_pass_filtering.band'
        bandpass_tr = 'parameters.connectivity.band_pass_filtering.tr'
        b_pca = 'parameters.connectivity.pca_transform.apply'
        pca = 'parameters.connectivity.pca_transform.components'

        # Define parameters
        d['participant_id'] = participant_id
        d['compress'] = self.get(compress, False)
        d['low_variance_correction'] = self.get(b_lve, False)
        d['low_variance_in_seed'] = self.get(lve_seed)
        d['low_variance_in_target'] = self.get(lve_target)
        d['arctanh_transform'] = self.get(arctanh, False)

        if session:
            d['session_id'] = self.wildcard('str(wildcards.session)')

        else:
            # For multisession, PCA transform is applied in merge_sessions
            if self.get(b_pca, False):
                d['pca_transform'] = self.get(pca)
            else:
                d['pca_transform'] = False

        if self.get(confounds_file, None):
            delimiter = repr(self.get(confounds_sep))
            delimiter = delimiter.replace('\'', '')
            d['confounds_delimiter'] = delimiter
            d['confounds_columns'] = self.get(confounds_cols)

        if self.get(b_smoothing, False):
            d['smoothing'] = self.get(smoothing)
        else:
            d['smoothing'] = False

        if self.get(b_bandpass, False):
            d['bandpass'] = (self.get(bandpass_band), self.get(bandpass_tr))
        else:
            d['bandpass'] = False

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params, log)' % self.name


class RuleMergeSessions(BaseRule):
    name = 'merge_sessions'
    dependencies = {'has_sessions': True}

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        session = self.get('data.session', None)
        return True if session is not None else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        file = 'individual/{participant_id}/connectivity_{session}.npz'
        sessions = self.get('data.session')
        sessions = 'session=%s' % sessions
        participant_id = 'participant_id=\'{participant_id}\''

        # Define parameters
        d['sessions'] = self.fwrap([file, sessions, participant_id], 'expand')

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        connectivity = 'individual/{participant_id}/connectivity.npz'

        # Define parameters
        d['connectivity'] = connectivity

        return d

    @property
    def log(self):
        return 's:log/{participant_id}.%s.log' % self.name

    @property
    def benchmark(self):
        return 's:benchmarks/{participant_id}.%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        wildcards = '{wildcards.participant_id}'
        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        compress = 'parameters.report.compress_output'
        b_pca = 'parameters.connectivity.pca_transform.apply'
        pca = 'parameters.connectivity.pca_transform.components'

        # Define parameters
        d['compress'] = self.get(compress, False)

        if self.get(b_pca, False):
            d['pca_transform'] = self.get(pca)
        else:
            d['pca_transform'] = False

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params, log)' % self.name


class RuleKMeansClustering(BaseRule):
    name = 'kmeans_clustering'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self) -> bool:
        cluster_method = self.get('parameters.clustering.method')
        return True if cluster_method == 'kmeans' else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        conn = self.get('data.connectivity', None)
        conn_default = 'individual/{participant_id}/connectivity.npz'

        # Define parameters
        d['connectivity'] = conn if conn else conn_default

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        labels = 'individual/{participant_id}/{n_clusters}cluster_labels.npy'

        # Define parameters
        d['labels'] = labels

        return d

    @property
    def log(self):
        wildcards = '{participant_id}.k{n_clusters}'
        return 's:log/%s.%s.log' % (wildcards, self.name)

    @property
    def benchmark(self):
        wildcards = '{participant_id}.k{n_clusters}'
        return 's:benchmarks/%s.%s.log' % (wildcards, self.name)

    @property
    def cluster_json(self):
        d = dict()
        wildcards = '{wildcards.participant_id}.k{wildcards.n_clusters}'
        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def resources(self):
        d = dict()

        # Parameter keys & files
        mem_mb = 'mem_mb.clustering'

        # Define parameters
        d['mem_mb'] = self.get(mem_mb, 1000)

        return d

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        algorithm = 'parameters.clustering.cluster_options.algorithm'
        init = 'parameters.clustering.cluster_options.init'
        max_iter = 'parameters.clustering.cluster_options.max_iter'
        n_init = 'parameters.clustering.cluster_options.n_init'
        n_clusters = self.wildcard('int(wildcards.n_clusters)')

        # Define parameters
        d['algorithm'] = self.get(algorithm)
        d['init'] = self.get(init)
        d['max_iter'] = self.get(max_iter)
        d['n_init'] = self.get(n_init)
        d['n_clusters'] = n_clusters

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params, log)' % self.name


class RuleSpectralClustering(BaseRule):
    name = 'spectral_clustering'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self) -> bool:
        cluster_method = self.get('parameters.clustering.method')
        return True if cluster_method == 'spectral' else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        conn = self.get('data.connectivity', None)
        conn_default = 'individual/{participant_id}/connectivity.npz'

        # Define parameters
        d['connectivity'] = conn if conn else conn_default

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        labels = 'individual/{participant_id}/{n_clusters}cluster_labels.npy'

        # Define parameters
        d['labels'] = labels

        return d

    @property
    def log(self):
        wildcards = '{participant_id}.k{n_clusters}'
        return 's:log/%s.%s.log' % (wildcards, self.name)

    @property
    def benchmark(self):
        return 's:benchmarks/{participant_id}.k{n_clusters}.%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        wildcards = '{wildcards.participant_id}.k{wildcards.n_clusters}'
        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def resources(self):
        d = dict()

        # Parameter keys & files
        mem_mb = 'mem_mb.clustering'

        # Define parameters
        d['mem_mb'] = self.get(mem_mb, 1000)

        return d

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        n_init = 'parameters.clustering.cluster_options.n_init'
        kernel = 'parameters.clustering.cluster_options.kernel'
        gamma = 'parameters.clustering.cluster_options.gamma'
        n_neighbors = 'parameters.clustering.cluster_options.n_neighbors'
        assign_labels = 'parameters.clustering.cluster_options.assign_labels'
        degree = 'parameters.clustering.cluster_options.degree'
        coef0 = 'parameters.clustering.cluster_options.coef0'
        eigen_tol = 'parameters.clustering.cluster_options.eigen_tol'
        eigen_solver = 'parameters.clustering.cluster_options.eigen_solver'
        n_clusters = self.wildcard('int(wildcards.n_clusters)')

        kernel = self.get(kernel)
        eigen_solver = self.get(eigen_solver)

        # Define parameters
        d['n_init'] = self.get(n_init)
        d['kernel'] = kernel
        d['assign_labels'] = self.get(assign_labels)
        d['eigen_solver'] = eigen_solver
        d['n_clusters'] = n_clusters

        if kernel in ('rbf', 'polynomial', 'sigmoid', 'laplacian', 'chi2'):
            d['gamma'] = self.get(gamma, None)

        if kernel == 'nearest_neighbors':
            d['n_neighbors'] = self.get(n_neighbors)

        if kernel == 'polynomial':
            d['degree'] = self.get(degree)

        if kernel in ('polynomial', 'sigmoid'):
            d['coef0'] = self.get(coef0)

        if eigen_solver == 'arpack':
            d['eigen_tol'] = self.get(eigen_tol)

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params, log)' % self.name


class RuleAgglomerativeClustering(BaseRule):
    name = 'agglomerative_clustering'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self) -> bool:
        cluster_method = self.get('parameters.clustering.method')
        return True if cluster_method == 'agglomerative' else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        conn = self.get('data.connectivity', None)
        conn_default = 'individual/{participant_id}/connectivity.npz'

        # Define parameters
        d['connectivity'] = conn if conn else conn_default

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        labels = 'individual/{participant_id}/{n_clusters}cluster_labels.npy'

        # Define parameters
        d['labels'] = labels

        return d

    @property
    def log(self):
        wildcards = '{participant_id}.k{n_clusters}'
        return 's:log/%s.%s.log' % (wildcards, self.name)

    @property
    def benchmark(self):
        wildcards = '{participant_id}.k{n_clusters}'
        return 's:benchmarks/%s.%s.log' % (wildcards, self.name)

    @property
    def cluster_json(self):
        d = dict()
        wildcards = '{wildcards.participant_id}.k{wildcards.n_clusters}'
        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def resources(self):
        d = dict()

        # Parameter keys & files
        mem_mb = 'mem_mb.clustering'

        # Define parameters
        d['mem_mb'] = self.get(mem_mb, 1000)

        return d

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        distance = 'parameters.clustering.cluster_options.distance_metric'
        linkage = 'parameters.clustering.cluster_options.linkage'
        n_clusters = self.wildcard('int(wildcards.n_clusters)')

        # Define parameters
        d['distance_metric'] = self.get(distance)
        d['linkage'] = self.get(linkage)
        d['n_clusters'] = n_clusters

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params, log)' % self.name


class RuleValidateClusterLabels(BaseRule):
    name = 'validate_cluster_labels'

    def __init__(self, conf):
        super().__init__(conf)

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        participants = 'participants.tsv'
        labels = 'individual/{participant_id}/{n_clusters}cluster_labels.npy'
        ppid = 'participant_id=participants'
        n_clusters = self.get('parameters.clustering.n_clusters')
        n_clusters = 'n_clusters=%s' % n_clusters

        # Define parameters
        d['participants'] = participants
        d['labels'] = self.fwrap([labels, n_clusters, ppid], 'expand')
        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        touchfile = 'individual/.touchfile'

        # Define parameters
        d['touchfile'] = touchfile

        return d

    @property
    def log(self):
        return 's:log/%s.log' % self.name

    @property
    def benchmark(self):
        return 's:benchmarks/%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        d['out'] = 'log/{rule}-%s.out' % self.jid
        return d

    @property
    def threads(self):
        return 1

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        space = self.get('data.masks.space', 'standard')
        n_clusters = 'parameters.clustering.n_clusters'
        connectivity = 'individual/{participant_id}/connectivity.npz'
        labels = 'individual/{participant_id}/{n_clusters}cluster_labels.npy'

        # Define parameters
        d['connectivity'] = self.wildcard(repr(connectivity))
        d['labels'] = self.wildcard(repr(labels))
        d['n_clusters'] = self.get(n_clusters)
        d['is_native'] = True if space == 'native' else False

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params, log)' % self.name


class RuleGroupLevelClustering(BaseRule):
    name = 'group_level_clustering'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        space = self.get('data.masks.space', 'standard')
        return True if space == 'standard' else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        modality = self.get('modality')
        coords = self.get('data.seed_coordinates', None)
        coords_default = 'seed_coordinates.npy'
        labels = 'individual/{participant_id}/{n_clusters}cluster_labels.npy'
        ppid = 'participant_id=participants'
        n_clusters = 'n_clusters=\'{n_clusters}\''
        participants = 'participants.tsv'
        seed_img = 'seed_mask.%s' % self.nifti_ext
        touchfile = 'individual/.touchfile'

        # Define parameters
        d['seed_img'] = seed_img
        d['participants'] = participants
        d['seed_coordinates'] = coords if coords else coords_default
        d['labels'] = self.fwrap([labels, ppid, n_clusters], 'expand')
        d['touchfile'] = touchfile

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        labels = 'group/{n_clusters}clusters/labels.npz'
        img = 'group/{n_clusters}clusters/labeled_roi.%s' % self.nifti_ext

        # Define parameters
        d['group_labels'] = labels
        d['group_img'] = img

        return d

    @property
    def benchmark(self):
        return 's:benchmarks/k{n_clusters}.%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        wildcards = 'k{wildcards.n_clusters}'
        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def resources(self):
        d = dict()

        # Parameter keys & files
        mem_mb = 'mem_mb.clustering'

        # Define parameters
        d['mem_mb'] = self.get(mem_mb, 1000)

        return d

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        linkage = 'parameters.clustering.grouping.linkage'
        method = 'parameters.clustering.grouping.method'

        # Define parameters
        d['linkage'] = self.get(linkage)
        d['method'] = self.get(method)

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params)' % self.name


class RuleInternalValidity(BaseRule):
    name = 'internal_validity'

    def __init__(self, conf):
        super().__init__(conf)

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        modality = self.get('modality')
        conn = self.get('data.connectivity', None)
        conn_default = 'individual/{participant_id}/connectivity.npz'
        labels = 'individual/{participant_id}/{n_clusters}cluster_labels.npy'
        n_clusters = self.get('parameters.clustering.n_clusters')
        n_clusters = 'n_clusters=%s' % n_clusters
        ppid = 'participant_id=\'{participant_id}\''
        touchfile = 'individual/.touchfile'

        # Define parameters
        d['connectivity'] = conn if conn else conn_default
        d['labels'] = self.fwrap([labels, n_clusters, ppid], 'expand')
        d['touchfile'] = touchfile

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        scores = 'individual/{participant_id}/internal_validity.tsv'

        # Define parameters
        d['scores'] = self.fwrap(scores, 'temp')

        return d

    @property
    def benchmark(self):
        return 's:benchmarks/{participant_id}.%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        wildcards = '{wildcards.participant_id}'
        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def resources(self):
        d = dict()

        # Parameter keys & files
        mem_mb = 'mem_mb.clustering'

        # Define parameters
        d['mem_mb'] = self.get(mem_mb, 1000)

        return d

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        participant_id = self.wildcard('str(wildcards.participant_id)')
        metrics = 'parameters.clustering.validity.internal'

        # Define parameters
        d['participant_id'] = participant_id
        d['metrics'] = self.get(metrics)

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params)' % self.name


class RuleMergeInternalValidity(BaseRule):
    name = 'merge_internal_validity'

    def __init__(self, conf):
        super().__init__(conf)

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        participants = 'participants.tsv'
        validity = 'individual/{participant_id}/internal_validity.tsv'
        ppid = 'participant_id=participants'
        n_clusters = self.get('parameters.clustering.n_clusters')
        n_clusters = 'n_clusters=%s' % n_clusters

        # Define parameters
        d['participants'] = participants
        d['validity'] = self.fwrap([validity, ppid, n_clusters], 'expand')

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        scores = 'individual/internal_validity.tsv'

        # Define parameters
        d['scores'] = scores

        return d

    @property
    def benchmark(self):
        return 's:benchmarks/%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        d['out'] = 'log/{rule}-%s.out' % self.jid
        return d

    @property
    def threads(self):
        return 1

    @property
    def run(self):
        return 'tasks.%s(input, output)' % self.name


class RuleIndividualSimilarity(BaseRule):
    name = 'individual_similarity'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        space = self.get('data.masks.space', 'standard')
        return True if space == 'standard' else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        labels = 'group/{n_clusters}clusters/labels.npz'

        # Define parameters
        d['labels'] = labels

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        similarity = 'group/{n_clusters}clusters/individual_similarity.npy'

        # Define parameters
        d['individual_similarity_matrix'] = similarity

        return d

    @property
    def benchmark(self):
        return 's:benchmarks/k{n_clusters}.%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        wildcards = 'k{wildcards.n_clusters}'
        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        metric = 'parameters.clustering.validity.similarity'

        # Define parameters
        d['metric'] = self.get(metric)

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params)' % self.name


class RuleGroupSimilarity(BaseRule):
    name = 'group_similarity'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        space = self.get('data.masks.space', 'standard')
        return True if space == 'standard' else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        participants = 'participants.tsv'
        labels = 'group/{n_clusters}clusters/labels.npz'
        n_clusters = self.get('parameters.clustering.n_clusters')
        n_clusters = 'n_clusters=%s' % n_clusters

        # Define parameters
        d['participants'] = participants
        d['labels'] = self.fwrap([labels, n_clusters], 'expand')

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        similarity = 'group/group_similarity.tsv'
        cophenet = 'group/cophenetic_correlation.tsv'

        # Define parameters
        d['group_similarity'] = similarity
        d['cophenetic_correlation'] = cophenet

        return d

    @property
    def benchmark(self):
        return 's:benchmarks/%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        d['out'] = 'log/{rule}-%s.out' % self.jid
        return d

    @property
    def threads(self):
        return 1

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        metric = 'parameters.clustering.validity.similarity'

        # Define parameters
        d['metric'] = self.get(metric)

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params)' % self.name


class RulePlotInternalValidity(BaseRule):
    name = 'plot_internal_validity'

    def __init__(self, conf):
        super().__init__(conf)

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        intval = 'individual/internal_validity.tsv'

        # Define parameters
        d['internal_validity'] = intval

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        figformat = self.get('parameters.report.figure_format')
        plot = 'individual/internal_validity_{metric}.%s' % figformat

        # Define parameters
        d['figure'] = plot

        return d

    @property
    def benchmark(self):
        return 's:benchmarks/{metric}.%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        wildcards = '{wildcards.metric}'
        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        metric = self.wildcard('str(wildcards.metric)')
        figformat = 'parameters.report.figure_format'

        # Define parameters
        d['metric'] = metric
        d['figure_format'] = self.get(figformat)

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params)' % self.name


class RulePlotIndividualSimilarity(BaseRule):
    name = 'plot_individual_similarity'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        space = self.get('data.masks.space', 'standard')
        return True if space == 'standard' else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        similarity = 'group/{n_clusters}clusters/individual_similarity.npy'

        # Define parameters
        d['individual_similarity'] = similarity

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        path = 'group/{n_clusters}clusters'
        heatmap = 'individual_similarity_heatmap.png'
        clustermap = 'individual_similarity_clustermap.png'

        # Define parameters
        d['heatmap'] = opj(path, heatmap)
        d['clustermap'] = opj(path, clustermap)

        return d

    @property
    def benchmark(self):
        return 's:benchmarks/k{n_clusters}.%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        wildcards = 'k{wildcards.n_clusters}'
        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def run(self):
        return 'tasks.%s(input, output)' % self.name


class RulePlotGroupSimilarity(BaseRule):
    name = 'plot_group_similarity'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        space = self.get('data.masks.space', 'standard')
        return True if space == 'standard' else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        groupsim = 'group/group_similarity.tsv'
        cophenet = 'group/cophenetic_correlation.tsv'

        # Define parameters
        d['group_similarity'] = groupsim
        d['cophenetic_correlation'] = cophenet

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        figformat = self.get('parameters.report.figure_format')
        similarity = 'group/group_similarity.%s' % figformat
        accuracy = 'group/relabeling_accuracy.%s' % figformat
        cophenet = 'group/cophenetic_correlation.%s' % figformat

        # Define parameters
        d['group_similarity'] = similarity
        d['relabel_accuracy'] = accuracy
        d['cophenetic_correlation'] = cophenet

        return d

    @property
    def benchmark(self):
        return 's:benchmarks/%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        d['out'] = 'log/{rule}-%s.out' % self.jid
        return d

    @property
    def threads(self):
        return 1

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        figformat = 'parameters.report.figure_format'

        # Define parameters
        d['figure_format'] = self.get(figformat)

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params)' % self.name


class RulePlotLabeledROI(BaseRule):
    name = 'plot_labeled_roi'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        space = self.get('data.masks.space', 'standard')
        return True if space == 'standard' else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        seed_img = 'seed_mask.%s' % self.nifti_ext
        labels = 'group/{n_clusters}clusters/labels.npz'

        # all labels files must exist so the reference can be loaded
        n_clusters = self.get('parameters.clustering.n_clusters')
        n_clusters = 'n_clusters=%s' % n_clusters
        reference = 'group/{n_clusters}clusters/labels.npz'

        # Define parameters
        d['seed_img'] = seed_img
        d['labels'] = labels
        d['reference'] = self.fwrap([reference, n_clusters], 'expand')

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        figformat = self.get('parameters.report.figure_format')
        figure = 'group/{n_clusters}clusters/voxel_plot_{view}.%s' % figformat

        # Define parameters
        d['figure'] = figure

        return d

    @property
    def benchmark(self):
        return 's:benchmarks/k{n_clusters}.{view}.%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        wildcards = 'k{wildcards.n_clusters}.{wildcards.view}'
        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        n_clusters = self.wildcard('int(wildcards.n_clusters)')
        all_clusters = 'parameters.clustering.n_clusters'
        view = self.wildcard('str(wildcards.view)')

        # Define parameters
        d['n_clusters'] = n_clusters
        d['all_clusters'] = self.get(all_clusters)
        d['view'] = view

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params)' % self.name


class RuleMergeIndividualLabels(BaseRule):
    name = 'merge_individual_labels'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        ind_plots = self.get('parameters.report.individual_plots', False)
        is_native = self.get('data.masks.space', 'standard') == 'native'
        return True if ind_plots or is_native else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        labels = 'individual/{{participant_id}}/{n_clusters}cluster_labels.npy'
        n_clusters = self.get('parameters.clustering.n_clusters')
        n_clusters = 'n_clusters=%s' % n_clusters
        touchfile = 'individual/.touchfile'

        # Define parameters
        d['labels'] = self.fwrap([labels, n_clusters], 'expand')
        d['touchfile'] = touchfile

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        merged_labels = 'individual/{participant_id}/cluster_labels.npz'

        # Define parameters
        d['merged_labels'] = merged_labels

        return d

    @property
    def benchmark(self):
        return 's:benchmarks/{participant_id}.%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        wildcards = '{wildcards.participant_id}'
        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def run(self):
        return 'tasks.%s(input, output)' % self.name


class RulePlotIndividualLabeledROI(BaseRule):
    name = 'plot_individual_labeled_roi'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        ind_plots = self.get('parameters.report.individual_plots', False)
        is_native = self.get('data.masks.space', 'standard') == 'native'
        return True if ind_plots or is_native else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        space = self.get('data.masks.space', 'standard')
        labels = 'individual/{participant_id}/cluster_labels.npz'
        touchfile = 'individual/.touchfile'

        if space == 'native':
            path = 'individual/{participant_id}'
            seed_img = opj(path, 'seed_mask.%s' % self.nifti_ext)
        else:  # if space == 'standard'
            seed_img = 'seed_mask.%s' % self.nifti_ext

        # Define parameters
        d['touchfile'] = touchfile
        d['labels'] = labels
        d['seed_img'] = seed_img

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        figformat = self.get('parameters.report.figure_format')
        path = 'individual/{participant_id}'
        figure = '{n_clusters}cluster_voxel_plot_{view}.%s' % figformat

        # Define parameters
        d['figure'] = opj(path, figure)

        return d

    @property
    def benchmark(self):
        wildcards = '{participant_id}.k{n_clusters}.{view}'
        return 's:benchmarks/%s.%s.log' % (wildcards, self.name)

    @property
    def cluster_json(self):
        d = dict()
        wildcards = 'k{wildcards.n_clusters}.{wildcards.view}'
        d['out'] = 'log/{rule}.%s-%s.out' % (wildcards, self.jid)
        return d

    @property
    def threads(self):
        return 1

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        view = self.wildcard('str(wildcards.view)')
        n_clusters = self.wildcard('int(wildcards.n_clusters)')
        all_clusters = self.get('parameters.clustering.n_clusters')
        participant_id = self.wildcard('str(wildcards.participant_id)')

        # Define parameters
        d['view'] = view
        d['n_clusters'] = n_clusters
        d['all_clusters'] = all_clusters
        d['participant_id'] = participant_id

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params)' % self.name


class RuleReferenceSimilarity(BaseRule):
    name = 'reference_similarity'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        space = self.get('data.masks.space', 'standard')
        references = self.get('data.references', False)
        return True if space == 'standard' and references else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        labels = 'group/{n_clusters}clusters/labels.npz'
        n_clusters = self.get('parameters.clustering.n_clusters')
        n_clusters = 'n_clusters=%s' % n_clusters
        refs = self.get('data.references', None)
        refpath = 'references'

        # Define parameters
        d['labels'] = self.fwrap([labels, n_clusters], 'expand')

        if isinstance(refs, list):
            d['references'] = [
                opj(refpath, opb(ref)) for ref in refs
            ]
        elif isinstance(refs, str):
            d['references'] = opj(refpath, opb(refs))

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        similarity = 'group/reference_similarity.tsv'

        # Define parameters
        d['similarity'] = similarity

        return d

    @property
    def benchmark(self):
        return 's:benchmarks/%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        d['out'] = 'log/{rule}-%s.out' % self.jid
        return d

    @property
    def threads(self):
        return 1

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        n_clusters = 'parameters.clustering.n_clusters'
        metric = 'parameters.clustering.validity.similarity'

        # Define parameters
        d['n_clusters'] = self.get(n_clusters)
        d['metric'] = self.get(metric)

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params)' % self.name


class RulePlotReferenceSimilarity(BaseRule):
    name = 'plot_reference_similarity'

    def __init__(self, conf):
        super().__init__(conf)

    def is_active(self):
        space = self.get('data.masks.space', 'standard')
        references = self.get('data.references', False)
        return True if space == 'standard' and references else False

    @property
    def input(self):
        d = dict()

        # Parameter keys & files
        reference_similarity = 'group/reference_similarity.tsv'

        # Define parameters
        d['reference_similarity'] = reference_similarity

        return d

    @property
    def output(self):
        d = dict()

        # Parameter keys & files
        figformat = self.get('parameters.report.figure_format')
        plot = 'group/reference_similarity.%s' % figformat

        # Define parameters
        d['figure'] = plot

        return d

    @property
    def benchmark(self):
        return 's:benchmarks/%s.log' % self.name

    @property
    def cluster_json(self):
        d = dict()
        d['out'] = 'log/{rule}-%s.out' % self.jid
        return d

    @property
    def threads(self):
        return 1

    @property
    def params(self):
        d = dict()

        # Parameter keys & files
        figformat = 'parameters.report.figure_format'
        metric = 'parameters.clustering.validity.similarity'

        # Define parameters
        d['figure_format'] = self.get(figformat)
        d['metric'] = self.get(metric)

        return d

    @property
    def run(self):
        return 'tasks.%s(input, output, params)' % self.name
