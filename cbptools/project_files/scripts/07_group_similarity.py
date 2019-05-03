#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics import adjusted_rand_score, v_measure_score, adjusted_mutual_info_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main(participants: str, labels_files: list, metric: str, output: dict, figure_format: str='png'):
    """Group Similarity (subject similarity to group clustering)

    Parameters
    ----------
    participants : str
        Path to the participant information dataframe file (as .tsv)
    labels_files : list
        List of file paths to the various group clustering files.
    metric : str, {'adjusted rand index', 'v measure', 'adjusted mutual information'}
        Name of the similarity metric used to generate the similarity scores
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

    participants = pd.read_csv(participants, sep='\t')['participant_id']
    data = pd.DataFrame()
    reference_data = pd.DataFrame()

    for file in labels_files:
        file = np.load(file)
        individual_labels = file['individual_labels']
        relabel_accuracy = file['relabel_accuracy']
        group_labels = file['group_labels']
        n_clusters = len(set(group_labels))
        reference_data = reference_data.append({
            'clusters': n_clusters,
            'cophenetic correlation': file['cophenetic_correlation']
        }, ignore_index=True)

        for participant_id, labels, accuracy in zip(participants, individual_labels, relabel_accuracy):
            data = data.append({
                'participant_id': participant_id,
                'clusters': n_clusters,
                'similarity': similarity(group_labels, labels),
                'relabel accuracy': accuracy
            }, ignore_index=True)

    data.to_csv(output.get('table1'), sep='\t', index=False)
    reference_data.to_csv(output.get('table2'), sep='\t', index=False)
    plt.ioff()

    # Generate Similarity Figure
    ax = sns.boxplot(x='clusters', y='similarity', data=data, showfliers=False)
    ax.set_title('Similarity of individual- to group-clusters')
    sns.despine(offset=10, trim=True)
    plt.savefig(output.get('figure1'), format=figure_format)
    plt.clf()

    # Generate Relabeling Accuracy Figure
    ax = sns.boxplot(x='clusters', y='relabel accuracy', data=data, showfliers=False)
    ax.set_title('Relabeling accuracy of individual- to group-level reference')
    sns.despine(offset=10, trim=True)
    plt.savefig(output.get('figure2'), format=figure_format)
    plt.clf()

    # Generate Cophenetic Correlation Figure
    ax = sns.pointplot(x='clusters', y='cophenetic correlation', data=reference_data)
    ax.set_title('Cophenetic Correlation of Group-level Clustering')
    sns.despine(offset=10, trim=True)
    plt.savefig(output.get('figure3'), format=figure_format)


if __name__ == '__main__':
    main(
        participants=snakemake.input.get('participants'),
        labels_files=snakemake.input.get('labels'),
        metric=snakemake.params.get('metric'),
        output={
            'table1': snakemake.output.get('table1'),
            'table2': snakemake.output.get('table2'),
            'figure1': snakemake.output.get('figure1'),
            'figure2': snakemake.output.get('figure2'),
            'figure3': snakemake.output.get('figure3')
        },
        figure_format=snakemake.params.get('figure_format')
    )

