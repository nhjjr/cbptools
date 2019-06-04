from cbptools.cluster import davies_bouldin_score, \
    weak_deletion_stability_score
from cbptools.utils import sort_files
from sklearn.metrics import silhouette_score, calinski_harabasz_score, \
    adjusted_rand_score, v_measure_score, adjusted_mutual_info_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np
import seaborn as sns


def internal_validity(connectivity: str, labels: list, participant_id: str,
                      metrics: list, out: str) -> None:
    """ Give single-participant connectivity and associated cluster
    label files for each requested number of clusters as input and
    this script returns the requested validity indices for each set
    of labels.

    Parameters
    ----------
    connectivity : str
        Path to the connectivity matrix
    labels : list
        Paths to the cluster label files of a particular subject
    participant_id : str
        Unique identifier for the current participant being processed.
    metrics : list
        List of the metrics that should be computed. ['silhouette',
        'davies-bouldin', 'calinski-harabasz', 'weak deletion
        stability'].
    out : str
        Output path for a spreadsheet of the various internal validity
        metrics
    """

    df = pd.DataFrame(columns=['participant_id', 'n_clusters'] + metrics)
    connectivity = np.load(connectivity)

    for label in labels:
        label = np.load(label) + 1
        df = df.append({
            'participant_id': participant_id,
            'n_clusters': len(set(label))
        }, ignore_index=True)
        idx = df.iloc[-1].name

        if 'silhouette_score' in metrics:
            df.loc[idx, 'silhouette_score'] = silhouette_score(
                connectivity, label,
                metric='euclidean'
            )

        if 'davies_bouldin_score' in metrics:
            df.loc[idx, 'davies_bouldin_score'] = davies_bouldin_score(
                connectivity,
                label
            )

        if 'calinski_harabasz_score' in metrics:
            df.loc[idx, 'calinski_harabasz_score'] = calinski_harabasz_score(
                connectivity,
                label
            )

        if 'weak deletion stability' in metrics:
            df.loc[idx, 'weak_deletion_stability_score'] = \
                weak_deletion_stability_score(
                    connectivity,
                    label
                )

    df.to_csv(out, sep='\t', index=False)


def summary_internal_validity(participants: str, validity: list,
                              metrics: list, out_table: str, out_figure: str,
                              figure_format: str = 'png') -> None:
    """ Generate a summary of the internal validity results.

    This script merges internal cluster validity into one table and
    generates a figure for a summary viewing.

    Parameters
    ----------
    participants : str
        Path to the participant information dataframe file
        (as .tsv)
    validity : list
        Paths to the various internal validity metric reports
        (as .tsv)
    metrics: list
        List of metrics that can be found in the validity metric
        reports
    out_table : str
        Output file path for tabular results
    out_figure : str
        Output file path for figure results
    figure_format : str, optional, {'png', 'svg', 'pdf', 'ps', 'eps'}
        Format of the figures that will be saved to disk
    """

    if not metrics:
        raise ValueError('Internal validity metrics must be set')

    # Merge internal validity scores by metric into subject x k
    # (n_clusters) tables
    validity = sort_files(
        participants,
        validity,
        pos=-1,
        sep='_',
        index_col='participant_id'
    )
    data = pd.concat((pd.read_csv(f, sep='\t', index_col=False)
                      for f in validity), ignore_index=True)
    data.to_csv(out_table, sep='\t', index=False)
    data.rename(columns={'n_clusters': 'clusters'}, inplace=True)

    # Generate validity metric figure
    plt.ioff()
    sns.set(style="whitegrid")
    n_cols = len(metrics)
    fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(4*n_cols, 4))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=1.5, hspace=None)

    for i, (metric, ax) in enumerate(zip(metrics, axes.flat[0:])):
        sns.boxplot(
            x='clusters',
            y=metric,
            data=data,
            ax=ax,
            showfliers=False,
            saturation=0.6,
            width=.8,
            dodge=True,
            linewidth=.75
        )
        ylabel = 'score' if i == 0 else ''
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel('')
        title = metric.replace("_", " ").title()
        ax.set_title(title, weight='bold').set_fontsize('10')
        ax.tick_params(axis='both', which='major', labelsize=8)

    fig.text(0.5, 0.04, 'clusters', ha='center', fontsize=10)
    sns.despine(offset=10, trim=True)
    plt.savefig(out_figure, format=figure_format)


def individual_similarity(labels: str, metric: str, n_clusters: str,
                          out_matrix: str, out_figure1: str, out_figure2: str,
                          figure_format: str = 'png') -> None:
    """Pairwise similarity matrix between all participant clustering results

    Parameters
    ----------
    labels : str
        Path to the group cluster labels, which contains the merged
        individual labels file used in this script.
    metric : str, {'adjusted rand index', 'v measure', 'adjusted mutual
        information'}. Name of the similarity metric used to generate
        the similarity scores
    n_clusters : str
        Number of clusters defined within the current set of labels
    out_matrix : str
        Output file path for the similarity matrix
    out_figure1 : str
        Output file path for figure 1 (unordered similarity matrix)
    out_figure2 : str
        Output file path for figure 2 (ordered similarity matrix)
    figure_format : str, optional, {'png', 'svg', 'pdf', 'ps', 'eps'}
        Format of the figures that will be saved to disk

    """

    metric = metric.lower()
    if metric == 'adjusted_rand_score':
        similarity = adjusted_rand_score

    elif metric == 'v_measure_score':
        similarity = v_measure_score

    elif metric == 'adjusted_mutual_info_score':
        similarity = adjusted_mutual_info_score

    else:
        raise ValueError('Metric \'{metric}\' not recognized')

    individual_labels = np.load(labels)['individual_labels']
    n_participants = individual_labels.shape[0]
    similarity_matrix = np.zeros((n_participants, n_participants))

    for (a_index, a), (b_index, b) in \
            itertools.combinations(enumerate(individual_labels), 2):
        similarity_matrix[a_index, b_index] = similarity(a, b)

    similarity_matrix += similarity_matrix.T
    np.fill_diagonal(similarity_matrix, 1)

    # save the similarity matrix to disk
    np.save(out_matrix, similarity_matrix)

    # Figure 2: Similarity Matrix (Heatmap)
    plt.ioff()
    ax = sns.heatmap(similarity_matrix, xticklabels=False, yticklabels=False)
    ax.set_title('Pairwise Similarity for n_clusters=%s (unordered)'
                 % n_clusters)
    plt.savefig(out_figure1, format=figure_format)
    plt.clf()

    # Figure 2: Similarity Matrix (clustermap)
    plt.ioff()
    sns.set(style="whitegrid")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=1.5, hspace=None)
    ax = sns.clustermap(
        similarity_matrix,
        metric='euclidean',
        method='average',
        robust=True,
        figsize=(16, 16),
        # **{'xticklabels': False, 'yticklabels': False}
    )
    ax.ax_heatmap.tick_params(left=False, bottom=False, right=False, top=False)
    plt.setp(ax.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
    plt.setp(ax.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
    ax.fig.suptitle('Pairwise Similarity for n_clusters=%s (clustermap)'
                    % n_clusters, weight='bold').set_fontsize('14')
    plt.savefig(out_figure2, format=figure_format)


def group_similarity(participants: str, labels_files: list, metric: str,
                     out_table1: str, out_table2: str, out_figure: str,
                     figure_format: str = 'png') -> None:
    """Group Similarity (subject similarity to group clustering)

    Parameters
    ----------
    participants : str
        Path to the participant information dataframe file (as .tsv)
    labels_files : list
        List of file paths to the various group clustering files.
    metric : str, {'adjusted rand index', 'v measure', 'adjusted
        mutual information'}. Name of the similarity metric used to
        generate the similarity scores
    out_table1 : str
        Output file path for table 1
    out_table2 : str
        Output file path for table 2
    out_figure : str
        Output file path for the group metrics figure
    figure_format : str, optional, {'png', 'svg', 'pdf', 'ps', 'eps'}
        Format of the figures that will be saved to disk

    """

    metric = metric.lower()
    if metric == 'adjusted_rand_score':
        similarity = adjusted_rand_score

    elif metric == 'v_measure_score':
        similarity = v_measure_score

    elif metric == 'adjusted_mutual_info_score':
        similarity = adjusted_mutual_info_score

    else:
        raise ValueError('Metric \'%s\' not recognized' % metric)

    participants = pd.read_csv(participants, sep='\t')['participant_id']
    data = pd.DataFrame(columns=['participant_id', 'clusters', 'similarity',
                                 'relabel accuracy'])
    reference_data = pd.DataFrame(columns=['clusters',
                                           'cophenetic correlation'])

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

        for participant_id, labels, accuracy \
                in zip(participants, individual_labels, relabel_accuracy):
            data = data.append({
                'participant_id': str(participant_id),
                'clusters': n_clusters,
                'similarity': similarity(group_labels, labels),
                'relabel accuracy': accuracy
            }, ignore_index=True)

    reference_data.clusters = reference_data.clusters.astype(int)
    data.clusters = data.clusters.astype(int)
    data.to_csv(out_table1, sep='\t', index=False)
    reference_data.to_csv(out_table2, sep='\t', index=False)

    # Generate Figure
    plt.ioff()
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=1.5, hspace=None)
    plt.suptitle('Individual- to Group-level cluster label comparisons',
                 fontsize=10, weight='bold')
    plot_options = {'showfliers': False, 'saturation': .6, 'width': .8,
                    'dodge': True, 'linewidth': .75}

    sns.boxplot(
        x='clusters',
        y='similarity',
        data=data,
        ax=ax[0],
        **plot_options
    )
    ax[0].set_ylabel(ax[0].get_ylabel(), fontsize=10)
    ax[0].set_xlabel('')
    ax[0].tick_params(labelsize=8)
    ax[0].set_title('Similarity', weight='bold', fontsize=10)

    sns.boxplot(
        x='clusters',
        y='relabel accuracy',
        data=data,
        ax=ax[1],
        **plot_options
    )
    ax[1].set_ylabel(ax[1].get_ylabel(), fontsize=10)
    ax[1].set_xlabel('')
    ax[1].tick_params(labelsize=8)
    ax[1].set_title('Relabeling Accuracy', weight='bold', fontsize=10)

    sns.pointplot(
        x='clusters',
        y='cophenetic correlation',
        data=reference_data,
        ax=ax[2]
    )
    ax[2].set_ylabel(ax[2].get_ylabel(), fontsize=10)
    ax[2].set_xlabel('')
    ax[2].tick_params(labelsize=8)
    ax[2].set_title('Cophenetic Correlation', weight='bold', fontsize=10)

    fig.text(0.5, 0.04, 'clusters', ha='center', fontsize=10)
    sns.despine(offset=10, trim=True)
    plt.savefig(out_figure, format=figure_format)
