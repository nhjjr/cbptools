from cbptools.cluster import relabel
from cbptools import plotting
import nibabel as nib
import numpy as np
import pandas as pd
import os


def plot_internal_validity(internal_validity: str, metrics: list, outdir: str,
                           figure_format: str = 'png') -> None:
    """ Generate a summary of the internal validity results.

    This script merges internal cluster validity into one table and
    generates a figure for a summary viewing.

    Parameters
    ----------
    internal_validity : str
        Path to the merged internal validity metric report (as .tsv)
    metrics : list
        List of metrics that can be found in the validity metric reports
    outdir : str
        Directory where the figures will be saved
    figure_format : str, optional, {'png', 'svg', 'pdf', 'ps', 'eps'}
        Format of the figures that will be saved to disk
    """

    if not metrics:
        raise ValueError('Internal validity metrics must be set')

    # Load data
    df = pd.read_csv(internal_validity, sep='\t')
    df.rename(columns={'n_clusters': 'clusters'}, inplace=True)

    # Generate validity metric figure
    for metric in metrics:
        fname = 'internal_validity_%s.%s' % (metric, figure_format)
        plotting.plot_scores(
            data=df[['clusters', metric]],
            x='clusters',
            y=metric,
            figure_format=figure_format,
            out_file=os.path.join(outdir, fname),
            source=internal_validity
        )


def plot_similarity(individual: str, group: str, cophenet: str, outdir: str,
                    figure_format: str = 'png') -> None:
    """Plot a pairwise similarity heatmap and clustermap, as well as group
    similarity box- and point plots.

    Parameters
    ----------
    individual : str
        Matrix of between-subject individual clustering similarity scores
    group : str
        Filepath to the group similarity scores
    cophenet : str
        Filepath to the cophenetic correlation scores
    outdir : str
        Output file path for the similarity matrix
    figure_format : str, optional, {'png', 'svg', 'pdf', 'ps', 'eps'}
        Format of the figures that will be saved to disk
    """

    # Individual Similarity
    individual_similarity = np.load(individual)
    for k, v in individual_similarity.items():
        # Heatmap
        out_file = os.path.join(outdir, '%s_heatmap.png' % k)
        plotting.plot_heatmap(v, out_file=out_file, source=individual,
                              plot_type='heatmap')

        # Clustermap
        out_file = os.path.join(outdir, '%s_clustermap.png' % k)
        plotting.plot_heatmap(v, out_file=out_file, source=individual,
                              plot_type='clustermap')

    # Group Similarity
    df = pd.read_csv(group, sep='\t')
    out_file = os.path.join(outdir, 'group_similarity.%s' % figure_format)
    plotting.plot_scores(df, x='clusters', y='similarity',
                         out_file=out_file, figure_format=figure_format,
                         source=group)

    # Relabel Accuracy
    out_file = os.path.join(outdir, 'relabeling_accuracy.%s' % figure_format)
    plotting.plot_scores(df, x='clusters', y='relabel accuracy',
                         out_file=out_file, figure_format=figure_format,
                         source=group)

    # Cophenetic Correlation
    df = pd.read_csv(cophenet, sep='\t')
    out_file = os.path.join(outdir, 'cophenetic_correlation.%s' % figure_format)
    plotting.plot_scores(df, x='clusters', y='cophenetic correlation',
                         out_file=out_file, figure_format=figure_format,
                         source=cophenet, plot_type='pointplot')


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
            plotting.plot_volumetric_roi(
                data=data,
                out_file=os.path.join(outdir, fname),
                view=view,
                facecolor='bright',
                edgecolor='dark'
            )
