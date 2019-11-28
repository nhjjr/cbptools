from cbptools.utils import sort_files
from cbptools.cluster import relabel
from cbptools.image import map_labels
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics.pairwise import kernel_metrics
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scipy import stats
import nibabel as nib
import numpy as np
import os


def kmeans_clustering(input: dict, output: dict, params: dict) -> None:
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
    """

    # Input, output, params
    connectivity_file = input.get('connectivity')
    labels_file = output.get('labels')
    algorithm = params.get('algorithm')
    init = params.get('init')
    max_iter = params.get('max_iter')
    n_init = params.get('n_init')
    n_clusters = params.get('n_clusters')

    _, ext = os.path.splitext(connectivity_file)
    connectivity = np.load(connectivity_file)

    if ext == '.npz':
        connectivity = connectivity.get('connectivity')

    # If the connectivity file is empty (connectivity could not be computed),
    # create an empty labels file
    if connectivity.size == 0:
        np.save(labels_file, np.array([]))
        return

    kwargs = {'algorithm': algorithm, 'init': init, 'max_iter': max_iter,
              'n_init': n_init}

    clustering = KMeans(n_clusters=n_clusters, **kwargs)
    clustering.fit(connectivity)
    labels = clustering.labels_

    # cluster labels are 0-indexed
    np.save(labels_file, labels)


def spectral_clustering(input: dict, output: dict, params: dict) -> None:
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
    """

    # Input, output, params
    connectivity_file = input.get('connectivity')
    labels_file = output.get('labels')
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

    _, ext = os.path.splitext(connectivity_file)
    connectivity = np.load(connectivity_file)

    if ext == '.npz':
        connectivity = connectivity.get('connectivity')

    # If the connectivity file is empty (connectivity could not be computed),
    # create an empty labels file
    if connectivity.size == 0:
        np.save(labels_file, np.array([]))
        return

    if isinstance(eigen_tol, str):
        eigen_tol = float(eigen_tol)

    if kernel not in kernel_metrics().keys():
        raise ValueError('Unknown kernel: %s' % kernel)

    kwargs = {'n_clusters': n_clusters, 'n_init': n_init, 'affinity': kernel,
              'assign_labels': assign_labels, 'eigen_solver': eigen_solver,
              'gamma': gamma, 'n_neighbors': n_neighbors, 'degree': degree,
              'coef0': coef0, 'eigen_tol': eigen_tol}

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    clustering = SpectralClustering(**kwargs)
    clustering.fit(connectivity)
    labels = clustering.labels_

    # cluster labels are 0-indexed
    np.save(labels_file, labels)


def agglomerative_clustering(input: dict, output: dict, params: dict) -> None:
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
    """

    # Input, output, params
    connectivity_file = input.get('connectivity')
    labels_file = output.get('labels')
    distance_metric = params.get('distance_metric')
    linkage = params.get('linkage')
    n_clusters = params.get('n_clusters')

    _, ext = os.path.splitext(connectivity_file)
    connectivity = np.load(connectivity_file)

    if ext == '.npz':
        connectivity = connectivity.get('connectivity')

    # If the connectivity file is empty (connectivity could not be computed),
    # create an empty labels file
    if connectivity.size == 0:
        np.save(labels_file, np.array([]))
        return

    kwargs = {'n_clusters': n_clusters, 'affinity': distance_metric,
              'linkage': linkage}

    clustering = AgglomerativeClustering(**kwargs)
    clustering.fit(connectivity)
    labels = clustering.labels_

    # cluster labels are 0-indexed
    np.save(labels_file, labels)


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
