from ..utils import sort_files, get_logger
from ..cluster import relabel
from ..image import map_labels
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics.pairwise import kernel_metrics
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from pathlib import Path
from scipy import stats
import nibabel as nib
import pandas as pd
import numpy as np
import logging
import os


def kmeans_clustering(input: dict, output: dict, params: dict,
                      log: list) -> None:
    """ Perform k-means clustering on the input connectivity matrix.

    Parameters
    ----------
    input : dict
        Input files, allowed: {connectivity}
    output : dict
        Output file, allowed {labels}
    params : dict
        The dict is equivalent to cluster_options in the CBPtools
        documentation on readthedocs.io under the parameters for 'clustering'.
    log : list
        Log files
    """

    # Input, output, params
    connectivity_file = input.get('connectivity')
    labels_file = output.get('labels')
    log_file = log[0]
    algorithm = params.get('algorithm')
    init = params.get('init')
    max_iter = params.get('max_iter')
    n_init = params.get('n_init')
    n_clusters = params.get('n_clusters')

    # Set up logging
    logger = get_logger('kmeans_clustering', log_file)

    _, ext = os.path.splitext(connectivity_file)
    connectivity = np.load(connectivity_file)

    if ext == '.npz':
        connectivity = connectivity.get('connectivity')

    # If the connectivity file is empty (connectivity could not be computed),
    # create an empty labels file
    if connectivity.size == 0:
        logger.warning('%s is empty, aborting clustering' % connectivity_file)
        np.save(labels_file, np.array([]))
        return

    kwargs = {'algorithm': algorithm, 'init': init, 'max_iter': max_iter,
              'n_init': n_init}

    debug_msg = str(['%s=%s' % (k, v) for k, v in kwargs.items()])
    debug_msg = debug_msg.strip('[]').replace('\'', '')
    logging.debug('clustering %s with options: %s'
                  % (connectivity_file, debug_msg))

    clustering = KMeans(n_clusters=n_clusters, **kwargs)
    clustering.fit(connectivity)
    labels = clustering.labels_

    # cluster labels are 0-indexed
    np.save(labels_file, labels)


def spectral_clustering(input: dict, output: dict, params: dict,
                        log: list) -> None:
    """ Perform spectral clustering on the input connectivity matrix.

    Parameters
    ----------
    input : dict
        Input files, allowed: {connectivity}
    output : dict
        Output file, allowed {labels}
    params : dict
        The dict is equivalent to cluster_options in the CBPtools
        documentation on readthedocs.io under the parameters for 'clustering'.
    log : dict
        Logging files, allowed {log}
    """

    # Input, output, params
    connectivity_file = input.get('connectivity')
    labels_file = output.get('labels')
    log_file = log[0]
    n_init = params.get('n_init')
    kernel = params.get('kernel')
    assign_labels = params.get('assign_labels')
    eigen_solver = params.get('eigen_solver')
    n_clusters = params.get('n_clusters')
    gamma = params.get('gamma', None)
    n_neighbors = params.get('n_neighbors', None)
    degree = params.get('degree', None)
    coef0 = params.get('coef0', None)
    eigen_tol = params.get('eigen_tol', None)

    # Set up logging
    logger = get_logger('spectral_clustering', log_file)

    _, ext = os.path.splitext(connectivity_file)
    connectivity = np.load(connectivity_file)

    if ext == '.npz':
        connectivity = connectivity.get('connectivity')

    # If the connectivity file is empty (connectivity could not be computed),
    # create an empty labels file
    if connectivity.size == 0:
        logger.warning('%s is empty, aborting clustering' % connectivity_file)
        np.save(labels_file, np.array([]))
        return

    if isinstance(eigen_tol, str):
        eigen_tol = float(eigen_tol)

    if kernel not in kernel_metrics().keys():
        msg = 'Unknown kernel (affinity): %s' % kernel
        logger.error(msg)
        raise ValueError(msg)

    gamma_kernels = ('rbf', 'polynomial', 'sigmoid', 'laplacian', 'chi2')
    if gamma is None and kernel in gamma_kernels:
        msg = 'Setting gamma to 1./%s (1./n_features)' % connectivity.shape[1]
        logger.warning(msg)
        gamma = 1./connectivity.shape[1]

    kwargs = {'n_clusters': n_clusters, 'n_init': n_init, 'affinity': kernel,
              'assign_labels': assign_labels, 'eigen_solver': eigen_solver,
              'gamma': gamma, 'n_neighbors': n_neighbors, 'degree': degree,
              'coef0': coef0}

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    debug_msg = str(['%s=%s' % (k, v) for k, v in kwargs.items()])
    debug_msg = debug_msg.strip('[]').replace('\'', '')
    logger.debug('clustering %s with options: %s'
                 % (connectivity_file, debug_msg))

    # Perform spectral clustering on the available tolerances
    try:
        kwargs['eigen_tol'] = eigen_tol
        clustering = SpectralClustering(**kwargs)
        clustering.fit(connectivity)
        labels = clustering.labels_

        if np.unique(labels).size != n_clusters:
            logging.error('%s: %s clusters requested, only %s found'
                          % (labels_file, n_clusters, np.unique(labels).size))
            np.save(labels_file, np.array([]))

        # cluster labels are 0-indexed
        np.save(labels_file, labels)

    except np.linalg.LinAlgError as exc:
        logger.error('%s: %s (try increasing the eigen_tol with arpack '
                     'as eigen_solver)' % (labels_file, exc))
        np.save(labels_file, np.array([]))


def agglomerative_clustering(input: dict, output: dict, params: dict,
                             log: list) -> None:
    """ Perform agglomerative clustering on the input connectivity matrix.

    Parameters
    ----------
    input : dict
        Input files, allowed: {connectivity}
    output : dict
        Output file, allowed {labels}
    params : dict
        The dict is equivalent to cluster_options in the CBPtools
        documentation on readthedocs.io under the parameters for 'clustering'.
    log : list
        Log files
    """

    # Input, output, params
    connectivity_file = input.get('connectivity')
    labels_file = output.get('labels')
    log_file = log[0]
    distance_metric = params.get('distance_metric')
    linkage = params.get('linkage')
    n_clusters = params.get('n_clusters')

    # Set up logging
    logger = get_logger('agglomerative_clustering', log_file)

    _, ext = os.path.splitext(connectivity_file)
    connectivity = np.load(connectivity_file)

    if ext == '.npz':
        connectivity = connectivity.get('connectivity')

    # If the connectivity file is empty (connectivity could not be computed),
    # create an empty labels file
    if connectivity.size == 0:
        logger.warning('%s is empty, aborting clustering' % connectivity_file)
        np.save(labels_file, np.array([]))
        return

    kwargs = {'n_clusters': n_clusters, 'affinity': distance_metric,
              'linkage': linkage}

    debug_msg = str(['%s=%s' % (k, v) for k, v in kwargs.items()])
    debug_msg = debug_msg.strip('[]').replace('\'', '')
    logging.debug('clustering %s with options: %s'
                  % (connectivity_file, debug_msg))

    clustering = AgglomerativeClustering(**kwargs)
    clustering.fit(connectivity)
    labels = clustering.labels_

    # cluster labels are 0-indexed
    np.save(labels_file, labels)


def validate_cluster_labels(input: dict, output: dict, params: dict,
                            log: list) -> None:
    """Ensure that all connectivity matrices could be computed"""

    # input, output, params
    labels_files = input.get('labels')
    participants_file = input.get('participants')
    log_file = log[0]
    touch_file = output.get('touchfile')
    connectivity_template = params.get('connectivity')
    labels_template = params.get('labels')
    n_clusters = params.get('n_clusters')
    is_native = params.get('is_native')

    # Set up logging
    logger = get_logger('validate_cluster_labels', log_file)

    bad_ppids = list()
    d_bad_ppids = dict()

    for labels_file in labels_files:
        ppid = labels_file.split('/')[1]
        labels = np.load(labels_file)

        if labels.size == 0 and ppid not in bad_ppids:
            logger.error('subject-id %s has problematic data '
                         '(check the connectivity and cluster logs)' % ppid)
            bad_ppids.append(ppid)

    if bad_ppids:
        logger.error('%s subject(s) with problematic data' % len(bad_ppids))

        for ppid in bad_ppids:
            c_file = connectivity_template.format(participant_id=ppid)
            l_files = [
                labels_template.format(participant_id=ppid, n_clusters=k)
                for k in n_clusters
            ]

            if os.path.exists(c_file):
                _, ext = os.path.splitext(c_file)
                r = np.load(c_file)

                if ext == '.npz':
                    r = r.get('connectivity')

                if r.size == 0:
                    reason = 'connectivity matrix for subject-id %s could ' \
                             'not be generated (check the connectivity ' \
                             'log(s) for this subject)' % ppid
                    logger.warning(reason)
                    logger.warning('removing output file %s' % c_file)
                    os.remove(c_file)
            else:
                d_bad_ppids[ppid] = 'connectivity matrix for subject-id %s ' \
                                    'is missing' % ppid
                logger.warning(d_bad_ppids[ppid])

            for file in l_files:
                if os.path.exists(file):
                    labels = np.load(file)

                    if labels.size == 0 and ppid not in d_bad_ppids.keys():
                        d_bad_ppids[ppid] = 'cluster labels for subject-id ' \
                                            '%s could not be generated ' \
                                            '(check the clustering log(s)' \
                                            'for this subject)' % ppid
                        logger.warning(d_bad_ppids[ppid])

            for file in l_files:
                if os.path.exists(file):
                    logger.warning('removing output file %s' % file)
                    os.remove(file)

        # Create a suggested participants file to proceed with the processing
        participants = pd.read_csv(participants_file, sep='\t')
        ppids = participants['participant_id']
        bad_ppids = list(set(bad_ppids))
        ppids = list(set(ppids) - set(bad_ppids))

        if (len(ppids) > 0 and is_native) or (len(ppids) > 1):
            suggested_file = 'participants_suggested.tsv'
            excluded_file = 'participants_excluded.tsv'
            df = pd.DataFrame(ppids, columns=['participant_id'])
            df.sort_values(by='participant_id', inplace=True)
            df.to_csv(suggested_file, sep='\t', index=False)

            data = [{'participant_id': k, 'reason': v}
                    for k, v in d_bad_ppids.items()]
            df = pd.DataFrame(data)
            df.sort_values(by='participant_id', inplace=True)
            df.to_csv(excluded_file, sep='\t', index=False)

            logger.info('created %s. To proceed without the problematic '
                        'subjects, replace %s with this file and run '
                        'snakemake again'
                        % (suggested_file, participants_file))

            logger.info('created %s as a reference for subjects with '
                        'problematic data and the reason why they are '
                        'excluded' % excluded_file)
        else:
            logger.info('not enough participants left to continue processing')

        raise ValueError(
            '%s subject(s) with problematic data. Read %s for more details'
            % (len(bad_ppids), log_file)
        )

    else:
        # Touch an output file that subsequent rules depend on
        logger.info('no problems found with the cluster labels')
        Path(touch_file).touch()


def group_level_clustering(input: dict, output: dict, params: dict) -> None:
    """ Perform group-level analysis on all individual participant
    clustering results.

    Parameters
    ----------
    input : dict
        Input files, allowed: {seed_img, participants, seed_coordinates,
        labels}
    output : dict
        Output file, allowed {group_labels, group_img}
    params : dict
        Parameters, allowed {linkage, method}. The options parameter is
        equivalent to grouping in the CBPtools documentation on readthedocs.io
        under the parameters for 'clustering'.

    """

    # Input, output, params
    participants = input.get('participants')
    labels = input.get('labels')
    out_labels = output.get('group_labels')
    out_img = output.get('group_img')
    method = params.get('method')
    linkage = params.get('linkage')
    seed_img = input.get('seed_img')
    seed_coordinates = input.get('seed_coordinates')

    if method not in ('agglomerative', 'mode'):
        raise ValueError('Unknown group cluster method: %s' % method)

    # Aggregate subject-level cluster labels into one matrix
    # Resulting shape is (participants, voxels)
    labels = sort_files(participants, labels, sep='/', pos=1)
    labels = np.asarray([np.load(f) for f in labels])

    if len(labels.shape) != 2:
        raise ValueError('Cluster label length mismatch between included '
                         'label files')

    # Hierarchical clustering on all labels
    x = labels.T
    y = pdist(x, metric='hamming')
    z = hierarchy.linkage(y, method=linkage, metric='hamming')
    cophenetic_correlation, *_ = hierarchy.cophenet(z, y)
    group_labels = hierarchy.cut_tree(z, n_clusters=len(np.unique(x)))
    group_labels = np.squeeze(group_labels)  # (N, 1) to (N,)

    # Use the hierarchical clustering as a reference to relabel individual
    # participant clustering results
    relabeled = np.empty((0, labels.shape[1]), int)
    accuracy = []

    # iterate over individual participant labels (rows)
    for label in labels:
        x, acc = relabel(reference=group_labels, x=label)
        relabeled = np.vstack([relabeled, x])
        accuracy.append(acc)

    labels = relabeled

    if method == 'agglomerative':
        np.savez(
            out_labels,
            individual_labels=labels,
            relabel_accuracy=accuracy,
            group_labels=group_labels,
            cophenetic_correlation=cophenetic_correlation,
            method='agglomerative'
        )

    elif method == 'mode':
        mode, count = stats.mode(labels, axis=0)
        np.savez(
            out_labels,
            individual_labels=labels,
            relabel_accuracy=accuracy,
            hierarchical_group_labels=group_labels,
            cophenetic_correlation=cophenetic_correlation,
            group_labels=np.squeeze(mode),
            mode_count=np.squeeze(count),
            method='mode'
        )

        # Set group labels to mode for mapping
        group_labels = np.squeeze(mode)

    # Map labels to seed-mask image based on indices
    seed_img = nib.load(seed_img)
    seed_indices = np.load(seed_coordinates)
    group_labels += 1  # avoid 0-labeling
    group_img = map_labels(
        img=seed_img,
        labels=group_labels,
        indices=seed_indices
    )
    nib.save(group_img, out_img)


def merge_individual_labels(input: dict, output: dict) -> None:
    """ Merge individual label results when no group analysis will be
    performed.

    Parameters
    ----------
    input : dict
        Input files, allowed: {labels}
    output : dict
        Output file, allowed {merged_labels}
    """

    # Input, output, params
    label_files = input.get('labels')
    merged_labels = output.get('merged_labels')

    all_labels = dict()

    for label_file in label_files:
        basename = os.path.basename(label_file)
        basename, _ = os.path.splitext(basename)
        labels = np.load(label_file)
        all_labels[basename] = labels

    np.savez(merged_labels, **all_labels)
