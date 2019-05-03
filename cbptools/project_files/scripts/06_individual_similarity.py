#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics import adjusted_rand_score, v_measure_score, adjusted_mutual_info_score
from scipy.cluster import hierarchy
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main(labels: str, metric: str, n_clusters: str, output: dict, figure_format: str='png'):
    """Pairwise similarity matrix between all participant clustering results

    Parameters
    ----------
    labels : str
        Path to the group cluster labels, which contains the merged individual labels file used in this script.
    metric : str, {'adjusted rand index', 'v measure', 'adjusted mutual information'}
        Name of the similarity metric used to generate the similarity scores
    n_clusters : str
        Number of clusters defined within the current set of labels
    output : dict
        Dictionary containing the various output file paths.
    figure_format : str, optional, {'png', 'svg', 'pdf', 'ps', 'eps'}
        Format of the figures that will be saved to disk

    """

    metric = metric.lower()
    if metric == 'adjusted rand index':
        similarity = adjusted_rand_score
    elif metric == 'v measure':
        similarity = v_measure_score
    elif metric == 'adjusted mutual information':
        similarity = adjusted_mutual_info_score
    else:
        raise ValueError('Metric \'{metric}\' not recognized')

    individual_labels = np.load(labels)['individual_labels']
    n_participants = individual_labels.shape[0]
    similarity_matrix = np.zeros((n_participants, n_participants))

    for (a_index, a), (b_index, b) in itertools.combinations(enumerate(individual_labels), 2):
        similarity_matrix[a_index, b_index] = similarity(a, b)

    similarity_matrix += similarity_matrix.T  # matrix is symmetrical
    np.fill_diagonal(similarity_matrix, 1)  # diagonal is self-similarity which is always 1

    # save the similarity matrix to disk
    np.save(output.get('matrix'), similarity_matrix)
    plt.ioff()

    # Figure 1: Unordered Similarity Matrix
    ax = sns.heatmap(similarity_matrix, xticklabels=False, yticklabels=False)
    ax.set_title(f'Pairwise Similarity for n_clusters={n_clusters} (unordered)')
    plt.savefig(output.get('figure1'), format=figure_format)
    plt.clf()

    # Figure 2: Similarity Matrix ordered by Dendrogram
    y = hierarchy.linkage(similarity_matrix, method='centroid')
    z = hierarchy.dendrogram(y, orientation='right', no_plot=True)
    index = z['leaves']
    similarity_matrix = similarity_matrix[index, :]
    similarity_matrix = similarity_matrix[:, index]
    ax = sns.heatmap(similarity_matrix, xticklabels=False, yticklabels=False)
    ax.set_title(f'Pairwise Similarity for n_clusters={n_clusters} (ordered)')
    plt.savefig(output.get('figure2'), format=figure_format)


if __name__ == '__main__':
    main(
        labels=snakemake.input.get('labels'),
        metric=snakemake.params.get('metric'),
        n_clusters=snakemake.wildcards.get('k'),
        output={
            'matrix': snakemake.output.get('matrix'),
            'figure1': snakemake.output.get('figure1'),
            'figure2': snakemake.output.get('figure2')
        },
        figure_format=snakemake.params.get('figure_format')
    )

