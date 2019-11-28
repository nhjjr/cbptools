from .. import plotting
from ..cluster import relabel
import nibabel as nib
import numpy as np
import pandas as pd


def plot_internal_validity(input: dict, output: dict, params: dict) -> None:
    """ Generate a summary of the internal validity results.

    This script merges internal cluster validity into one table and
    generates a figure for a summary viewing.

    Parameters
    ----------
    input : dict
        Input files, allowed: {internal_validity}
    output : dict
        Output files, allowed: {figure}
    params : dict
        Parameters, allowed {metric, figure_format}.
    """

    # input, params
    internal_validity_file = input.get('internal_validity')
    figure_file = output.get('figure')
    metric = params.get('metric')
    figure_format = params.get('figure_format')

    if not metric:
        raise ValueError('Internal validity metrics must be set')

    # Load data
    df = pd.read_csv(internal_validity_file, sep='\t')
    df.rename(columns={'n_clusters': 'clusters'}, inplace=True)

    # Generate validity metric figure
    plotting.plot_scores(
        data=df[['clusters', metric]],
        x='clusters',
        y=metric,
        figure_format=figure_format,
        out_file=figure_file,
        source=internal_validity_file
    )


def plot_individual_similarity(input: dict, output: dict) -> None:
    """Plot a pairwise similarity heatmap and clustermap.

    Parameters
    ----------
    input : dict
        Input files, allowed: {similarity}
    output : dict
        Output files, allowed: {heatmap, clustermap}
    """

    similarity_file = input.get('individual_similarity')
    heatmap_file = output.get('heatmap')
    clustermap_file = output.get('clustermap')
    data = np.load(similarity_file)

    plotting.plot_heatmap(data, out_file=heatmap_file, source=similarity_file,
                          plot_type='heatmap')
    plotting.plot_heatmap(data, out_file=clustermap_file,
                          source=similarity_file, plot_type='clustermap')


def plot_group_similarity(input: dict, output: dict, params: dict) -> None:
    """Plot a group similarity box- and point plots.

    Parameters
    ----------
    input : dict
        Input files, allowed: {individual_similarity, group_similarity,
        cophenetic_correlation}
    output : dict
        Output files, allowed: {group_similarity, relabel_accuracy,
        cophenetic_correlation}
    params : dict
        Parameters, allowed {figure_format}.
    """

    # input, params
    similarity_file = input.get('group_similarity')
    cophenet_file = input.get('cophenetic_correlation')
    similarity_figure = output.get('group_similarity')
    accuracy_figure = output.get('relabel_accuracy')
    cophenet_figure = output.get('cophenetic_correlation')
    figure_format = params.get('figure_format')
    sim_df = pd.read_csv(similarity_file, sep='\t')
    coph_df = pd.read_csv(cophenet_file, sep='\t')

    plotting.plot_scores(
        sim_df, x='clusters', y='similarity', out_file=similarity_figure,
        figure_format=figure_format, source=similarity_file
    )
    plotting.plot_scores(
        sim_df, x='clusters', y='relabel accuracy', out_file=accuracy_figure,
        figure_format=figure_format, source=similarity_file
    )
    plotting.plot_scores(
        coph_df, x='clusters', y='cophenetic correlation',
        out_file=cophenet_figure, figure_format=figure_format,
        source=cophenet_file, plot_type='pointplot'
    )


def plot_labeled_roi(input: dict, output: dict, params: dict) -> None:
    """ Relabel group results to most closely match in cluster-id allocation
    and then save a 3D volumetric voxel plot

    Parameters
    ----------
    input : dict
        Input files, allowed: {labels, seed_img}
    output : dict
        Output files, allowed: {figure}
    params : dict
        Parameters, allowed {n_clusters, all_clusters, view, figure_format}.
    """

    # input, params
    labels_file = input.get('labels')
    seed_img_file = input.get('seed_img')
    figure_file = output.get('figure')
    n_clusters = params.get('n_clusters')
    all_clusters = sorted(params.get('all_clusters'))
    view = params.get('view')

    seed_img = nib.load(seed_img_file)
    seed_data = seed_img.get_data()

    labels = np.load(labels_file)['group_labels']
    labels += 1  # prevent 0-indexing

    index = all_clusters.index(n_clusters)
    if index > 0:
        ref_clusters = all_clusters[index-1]
        ref_file = 'group/%sclusters/labels.npz' % ref_clusters
        reference = np.load(ref_file)['group_labels']
        reference += 1  # prevent 0-indexing
        labels, _ = relabel(reference, labels)

    data = np.zeros(seed_img.shape)
    data[np.where(seed_data > 0)] = labels

    plotting.plot_volumetric_roi(
        data=data, out_file=figure_file, view=view, facecolor='bright',
        edgecolor='dark')


def plot_individual_labeled_roi(input: dict, output: dict,
                                params: dict) -> None:
    """ Relabel group results to most closely match in cluster-id allocation
    and then save a 3D volumetric voxel plot

    Parameters
    ----------
    input : dict
        Input files, allowed: {labels, seed_img}
    output : dict
        Output files, allowed: {figure}
    params : dict
        Parameters, allowed {view, template, n_clusters, all_clusters}.
    """

    # input, params
    labels_file = input.get('labels')
    seed_img_file = input.get('seed_img')
    figure_file = output.get('figure')
    view = params.get('view')
    n_clusters = params.get('n_clusters')
    all_clusters = params.get('all_clusters')
    seed_img = nib.load(seed_img_file)
    seed_data = seed_img.get_data()
    template = '%scluster_labels' % n_clusters
    label_collection = np.load(labels_file)
    labels = label_collection.get(template)
    labels += 1  # prevent 0-indexing

    # Relabel to match the lower n_cluster labels
    index = all_clusters.index(n_clusters)
    if index > 0:
        ref_clusters = all_clusters[index-1]
        ref_template = '%scluster_labels' % ref_clusters
        reference = label_collection.get(ref_template)
        reference += 1  # prevent 0-indexing
        labels, _ = relabel(reference, labels)

    data = np.zeros(seed_img.shape)
    data[np.where(seed_data > 0)] = labels

    plotting.plot_volumetric_roi(
        data=data,
        out_file=figure_file,
        view=view,
        facecolor='bright',
        edgecolor='dark'
    )


def plot_reference_similarity(input: dict, output: dict, params: dict) -> None:
    """ Relabel group results to most closely match in cluster-id allocation
    and then save a 3D volumetric voxel plot

    Parameters
    ----------
    input : dict
        Input files, allowed: {reference_similarity}
    output : dict
        Output files, allowed: {figure}
    params : dict
        Parameters, allowed {figure_format}.
    """

    similarity_file = input.get('reference_similarity')
    figure_file = output.get('figure')
    figure_format = params.get('figure_format')
    metric = params.get('metric')
    data = pd.read_csv(similarity_file, sep='\t', index_col='reference')

    metric = metric.title()
    metric = metric.replace('_', ' ')

    plotting.plot_comparison(data, out_file=figure_file,
                             figure_format=figure_format,
                             source=similarity_file, title=metric)
