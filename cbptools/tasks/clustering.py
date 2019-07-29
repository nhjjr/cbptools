from cbptools.utils import sort_files
from cbptools.cluster import relabel
from cbptools.image import map_labels
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scipy import stats
import nibabel as nib
import numpy as np


def participant_level_clustering(connectivity, out: str, n_clusters: int,
                                 algorithm: str = 'auto',
                                 init: str = 'random', max_iter: int = 10000,
                                 n_init: int = 100) -> None:
    """ Perform k-means clustering on the input connectivity
    matrix.

    Parameters
    ----------
    connectivity : str
        Path to the connectivity matrix that should be clustered
        (.npy)
    out : str
        Output filename for the k-means labels (.npy)
    n_clusters : int
        The number of clusters to form. See
        sklearn.cluster.KMeans
    algorithm : str, optional
        K-means algorithm to use. See sklearn.cluster.KMeans
    init : str, optional
        Method for initialization, defaults to ‘random’.
        See sklearn.cluster.KMeans
    max_iter : int, optional
        Number of iterations of the k-means algorithm.
        See sklearn.cluster.KMeans
    n_init : int
        Number of initializations of the k-means algorithm.
        See sklearn.cluster.KMeans
    """

    connectivity = np.load(connectivity, allow_pickle=True)

    # If the connectivity file is empty (connectivity could not be computed),
    # create an empty labels file
    if connectivity.size == 0:
        np.save(out, np.array([]))
        return

    kmeans = KMeans(
        algorithm=algorithm,
        init=init,
        max_iter=max_iter,
        n_clusters=n_clusters,
        n_init=n_init
    )
    kmeans.fit(connectivity)

    # cluster labels are 0-indexed
    np.save(out, kmeans.labels_)


def group_level_clustering(seed_img: str, participants: str,
                           individual_labels: list, linkage: str,
                           out_labels: str, out_img: str,
                           method: str = 'mode',
                           seed_indices: str = None) -> None:
    """ Perform group-level analysis on all individual participant
    clustering results.

    Parameters
    ----------
    seed_img : str
        Path to the region-of-interest mask nifti image. This is
        used for projecting the cluster labels upon the
        region-of-interest mask.
    participants : str
        Path to the participants tsv file. This is used to order
        the group labels by participant_id.
    individual_labels : list
        Paths to all the participant clustering results. This is
        used as input for generating the group-clustering.
    linkage : str
        Linkage method to use for agglomerative clustering.
        Allowed values are: 'complete', 'average', 'single'.
        However, single is not recommended for this type of data.
    out_labels : str
        Output filename (.npz) for the relabeled individual participant
        labels, the relabeling accuracy, the hierarchical group labels,
        the cophenetic correlation for the hierarchical clustering, and
        if the method is 'mode' also the mode group labels and the counts
        for the mode.
    out_img : str
        Output filename for a nifti image in which the group-cluster
        labels are projected upon the region-of-interest
        mask.
    method : str
        Method defining the final group labels. Allowed values are:
        {'agglomerative', 'mode'}. For the agglomerative method, the
        group labels are defined as the labels obtained from hierarchical
        clustering. For the mode method, the individual participant
        labels are relabeled using the hierarchical clustering results
        as a reference. The mode is then taken from all relabeled
        participant clusterings and used as a group level clustering.
    seed_indices : str
        Path to the numpy file containing the indices of the seed voxels.
    """

    if method not in ('agglomerative', 'mode'):
        raise ValueError('Unknown group cluster method: %s' % method)

    # Aggregate subject-level cluster labels into one matrix
    # Resulting shape is (participants, voxels)
    individual_labels = sort_files(participants, individual_labels, pos=-1)
    individual_labels = np.asarray([np.load(f, allow_pickle=True)
                                    for f in individual_labels])

    if len(individual_labels.shape) != 2:
        raise ValueError('Cluster label length mismatch between included '
                         'label files')

    # Hierarchical clustering on all labels
    x = individual_labels.T
    y = pdist(x, metric='hamming')
    z = hierarchy.linkage(y, method=linkage, metric='hamming')
    cophenetic_correlation, *_ = hierarchy.cophenet(z, y)
    group_labels = hierarchy.cut_tree(z, n_clusters=len(np.unique(x)))
    group_labels = np.squeeze(group_labels)  # (N, 1) to (N,)

    # Use the hierarchical clustering as a reference to relabel individual
    # participant clustering results
    relabeled = np.empty((0, individual_labels.shape[1]), int)
    accuracy = []

    # iterate over individual participant labels (rows)
    for labels in individual_labels:
        x, acc = relabel(reference=group_labels, x=labels)
        relabeled = np.vstack([relabeled, x])
        accuracy.append(acc)

    individual_labels = relabeled

    if method == 'agglomerative':
        np.savez(
            out_labels,
            individual_labels=individual_labels,
            relabel_accuracy=accuracy,
            group_labels=group_labels,
            cophenetic_correlation=cophenetic_correlation,
            method='agglomerative'
        )

    elif method == 'mode':
        mode, count = stats.mode(individual_labels, axis=0)
        np.savez(
            out_labels,
            individual_labels=individual_labels,
            relabel_accuracy=accuracy,
            hierarchical_group_labels=group_labels,
            cophenetic_correlation=cophenetic_correlation,
            group_labels=np.squeeze(mode),
            mode_count=np.squeeze(count),
            method='mode'
        )

    # Map labels to seed-mask image based on indices
    seed_img = nib.load(seed_img)
    seed_indices = np.load(seed_indices, allow_pickle=True)
    group_labels += 1  # avoid 0-labeling
    group_img = map_labels(
        img=seed_img,
        labels=group_labels,
        indices=seed_indices
    )
    nib.save(group_img, out_img)
