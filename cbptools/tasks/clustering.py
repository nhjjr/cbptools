from cbptools.utils import sort_files
from cbptools.cluster import relabel
from cbptools.image import map_labels
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scipy import stats
import nibabel as nib
import numpy as np


def participant_level_clustering(connectivity, out: str, n_clusters: int, algorithm: str = 'auto',
                                 init: str = 'random', max_iter: int = 10000, n_init: int = 100):
    """ Perform k-means clustering on the input connectivity matrix.

    Parameters
    ----------
    connectivity : str
        Path to the connectivity matrix that should be clustered (.npy)
    out : str
        Output filename for the k-means labels (.npy)
    n_clusters : int
        The number of clusters to form. See sklearn.cluster.KMeans
    algorithm : str, optional
        K-means algorithm to use. See sklearn.cluster.KMeans
    init : str, optional
        Method for initialization, defaults to ‘random’. See sklearn.cluster.KMeans
    max_iter : int, optional
        Number of iterations of the k-means algorithm. See sklearn.cluster.KMeans
    n_init : int
        Number of initializations of the k-means algorithm. See sklearn.cluster.KMeans
    """

    connectivity = np.load(connectivity)

    # If the connectivity file is empty (connectivity could not be computed), create an empty labels file
    if connectivity.size == 0:
        np.save(out, np.array([]))
        return

    kmeans = KMeans(algorithm=algorithm, init=init, max_iter=max_iter, n_clusters=n_clusters, n_init=n_init)
    kmeans.fit(connectivity)
    np.save(out, kmeans.labels_)  # cluster labels are 0-indexed


def group_level_clustering(seed_img: str, participants: str, individual_labels: list, linkage: str,
                           input_data_type: str, out_labels: str, out_img: str, method: str = 'mode',
                           seed_indices: str = None):
    """ Perform group-level analysis on all individual participant clustering results.

    Parameters
    ----------
    seed_img : str
        Path to the region-of-interest mask nifti image. This is used for projecting the cluster labels upon the
        region-of-interest mask.
    participants : str
        Path to the participants tsv file. This is used to order the group labels by participant_id.
    individual_labels : list
        Paths to all the participant clustering results. This is used as input for generating the group-clustering.
    linkage : str
        Linkage method to use for agglomerative clustering. Allowed values are: 'complete', 'average', 'single'.
        However, single is not recommended for this type of data.
    input_data_type : str
        Modality that is being assessed. This is important for the order in which the unmasking takes place (i.e.,
        F order for 'dwi', and C order for 'func', due to how the masks are applied to the connectivity data).
    out_labels : str
        Output filename (.npz) for the relabeled individual participant labels, the relabeling accuracy, the
        hierarchical group labels, the cophenetic correlation for the hierarchical clustering, and if the method is
        'mode' also the mode group labels and the counts for the mode.
    out_img : str
        Output filename for a nifti image in which the group-cluster labels are projected upon the region-of-interest
        mask.
    method : str
        Method defining the final group labels. Allowed values are: {'agglomerative', 'mode'}. For the agglomerative
        method, the group labels are defined as the labels obtained from hierarchical clustering. For
        the mode method, the individual participant labels are relabeled using the hierarchical clustering results as
        a reference. The mode is then taken from all relabeled participant clusterings and used as a group level
        clustering.
    seed_indices : str
        Path to the numpy file containing the indices of the seed voxels.
    """

    methods = ('agglomerative', 'mode')

    if method not in methods:
        raise ValueError(f'Unknown group cluster method: {method}')

    # Aggregate subject-level cluster labels into one matrix
    individual_labels = sort_files(participants, individual_labels, pos=-1)  # sort files to match participant order
    individual_labels = np.asarray([np.load(f) for f in individual_labels])  # shape: participants by voxels

    if len(individual_labels.shape) != 2:
        raise ValueError('Cluster label length mismatch between included label files')

    # Hierarchical clustering on all labels
    x = individual_labels.T
    y = pdist(x, metric='hamming')
    z = hierarchy.linkage(y, method=linkage, metric='hamming')
    cophenetic_correlation, *_ = hierarchy.cophenet(z, y)
    group_labels = hierarchy.cut_tree(z, n_clusters=len(np.unique(x)))
    group_labels = np.squeeze(group_labels)  # (N, 1) to (N,)

    # Use the hierarchical clustering as a reference to relabel individual participant clustering results
    relabeled = np.empty((0, individual_labels.shape[1]), int)
    accuracy = []
    for labels in individual_labels:  # iterate over individual participant labels (rows)
        x, acc = relabel(reference=group_labels, x=labels)
        relabeled = np.vstack([relabeled, x])
        accuracy.append(acc)

    individual_labels = relabeled

    if method == 'agglomerative':
        np.savez(out_labels, individual_labels=individual_labels, relabel_accuracy=accuracy, group_labels=group_labels,
                 cophenetic_correlation=cophenetic_correlation, method='agglomerative')

    elif method == 'mode':
        mode, count = stats.mode(individual_labels, axis=0)
        np.savez(out_labels, individual_labels=individual_labels, relabel_accuracy=accuracy,
                 hierarchical_group_labels=group_labels, cophenetic_correlation=cophenetic_correlation,
                 group_labels=np.squeeze(mode), mode_count=np.squeeze(count), method='mode')

    # Map labels to seed-mask image based on indices
    if seed_indices:
        seed_indices = np.load(seed_indices)

    order = 'F' if input_data_type == 'dmri' else 'C'
    seed_img = nib.load(seed_img)
    group_labels += 1  # avoid 0-labeling
    group_img = map_labels(img=seed_img, labels=group_labels, seed_indices=seed_indices, order=order)
    nib.save(group_img, out_img)
