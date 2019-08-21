from cbptools.cluster import relabel
from cbptools.plotting import plot_volumetric_roi
import nibabel as nib
import numpy as np
import os


def plot_labeled_roi(group_labels: list, seed_img: str,
                     outdir: str, figure_format: str = 'png') -> None:
    """ Relabel group results to most closely match in cluster-id allocation
    and then save a 3D volumetric voxel plot

    Parameters
    ----------
    group_labels : list
        Paths to the group label files (clustering)
    seed_img : str
        Path to the region-of-interest mask nifti image.
    outdir : str
        Folder in which the figures will be stored
    figure_format : str, optional, {'png', 'svg', 'pdf', 'ps', 'eps'}
        Format of the figures that will be saved to disk
    """
    group_labels.sort()
    seed_img = nib.load(seed_img)
    seed_data = seed_img.get_data()

    for labels_file in group_labels:
        labels = np.load(labels_file)['group_labels']
        labels += 1  # 0-indexing

        if group_labels.index(labels_file) > 0:
            labels, _ = relabel(reference, labels)

        reference = labels
        data = np.zeros(seed_img.shape)
        data[np.where(seed_data > 0)] = labels

        views = ['right', 'left', 'superior', 'inferior', 'posterior',
                 'anterior']
        k = len(np.unique(labels))

        for view in views:
            fname = 'group_clustering_k%s_%s.%s' % (k, view, figure_format)
            plot_volumetric_roi(
                data=data,
                out_file=os.path.join(outdir, fname),
                view=view,
                facecolor='bright',
                edgecolor='dark'
            )
