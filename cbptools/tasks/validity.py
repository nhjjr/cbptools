from cbptools.cluster import davies_bouldin_score, find_centers, weak_deletion_stability, gap_score
from cbptools.utils import sort_files
from sklearn.metrics import silhouette_score, calinski_harabaz_score, adjusted_rand_score, v_measure_score, adjusted_mutual_info_score
from scipy.cluster import hierarchy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np
import seaborn as sns


def internal_validity(connectivity: str, labels: list, participant_id: str, metrics: list, out: str):
    """ Give single-participant connectivity and associated cluster label files for each requested number of clusters as
    input and this script returns the requested validity indices for each set of labels.

    Parameters
    ----------
    connectivity : str
        Path to the connectivity matrix
    labels : list
        Paths to the cluster label files of a particular subject
    participant_id : str
        Unique identifier for the current participant being processed.
    metrics : list
        List of the metrics that should be computed. ['silhouette', 'davies-bouldin', 'calinski-harabasz',
        'weak deletion stability'].
    out : str
        Output path for a spreadsheet of the various internal validity metrics
    """

    df = pd.DataFrame(columns=['participant_id', 'n_clusters'] + metrics)
    connectivity = np.load(connectivity)

    for label in labels:
        label = np.load(label) + 1
        df = df.append({'participant_id': participant_id, 'n_clusters': len(set(label))}, ignore_index=True)
        idx = df.iloc[-1].name

        if 'silhouette' in metrics:
            df.loc[idx, 'silhouette'] = silhouette_score(connectivity, label, metric='euclidean')

        if 'davies-bouldin' in metrics:
            df.loc[idx, 'davies-bouldin'] = davies_bouldin_score(connectivity, label)

        if 'calinski-harabasz' in metrics:
            df.loc[idx, 'calinski-harabasz'] = calinski_harabaz_score(connectivity, label)

        if 'weak deletion stability' in metrics:
            df.loc[idx, 'weak deletion stability'] = weak_deletion_stability(connectivity, label)

    df.to_csv(out, sep='\t', index=False)


def summary_internal_validity(participants: str, validity: list, internal_validity_metrics: list, out_table: str,
                              out_figure: str, figure_format: str = 'png'):
    """ Generate a summary of the internal validity results.

    This script merges internal cluster validity into one table and generates a figure for a summary viewing.

    Parameters
    ----------
    participants : str
        Path to the participant information dataframe file (as .tsv)
    validity : list
        Paths to the various internal validity metric reports (as .tsv)
    internal_validity_metrics: list
        List of metrics that can be found in the validity metric reports
    out_table : str
        Output file path for tabular results
    out_figure : str
        Output file path for figure results
    figure_format : str, optional, {'png', 'svg', 'pdf', 'ps', 'eps'}
        Format of the figures that will be saved to disk
    """

    # Merge internal validity scores by metric into subject x k (n_clusters) tables
    validity = sort_files(participants, validity, pos=-1, sep='_', index_col='participant_id')
    data = pd.concat((pd.read_csv(f, sep='\t', index_col=False) for f in validity), ignore_index=True)
    data.to_csv(out_table, sep='\t', index=False)

    # Generate validity metric figure
    plt.ioff()

    # Max column width = 3
    if len(internal_validity_metrics) > 3:
        ncols = 3
        nrows = int(np.floor(len(internal_validity_metrics)/3))+1
    else:
        ncols = len(internal_validity_metrics)
        nrows = 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

    for metric, ax in zip(internal_validity_metrics, axes.flat[0:]):
        data.rename(columns={'n_clusters': 'clusters'}, inplace=True)
        sns.boxplot(x='clusters', y=metric, data=data, ax=ax, showfliers=False)

    fig.suptitle('Internal Validity Scores')
    sns.despine(offset=10, trim=True)
    plt.savefig(out_figure, format=figure_format)


def individual_similarity(labels: str, metric: str, n_clusters: str, out_matrix: str, out_figure1: str,
                          out_figure2: str, figure_format: str = 'png'):
    """Pairwise similarity matrix between all participant clustering results

    Parameters
    ----------
    labels : str
        Path to the group cluster labels, which contains the merged individual labels file used in this script.
    metric : str, {'adjusted rand index', 'v measure', 'adjusted mutual information'}
        Name of the similarity metric used to generate the similarity scores
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
    np.save(out_matrix, similarity_matrix)
    plt.ioff()

    # Figure 1: Unordered Similarity Matrix
    ax = sns.heatmap(similarity_matrix, xticklabels=False, yticklabels=False)
    ax.set_title(f'Pairwise Similarity for n_clusters={n_clusters} (unordered)')
    plt.savefig(out_figure1, format=figure_format)
    plt.clf()

    # Figure 2: Similarity Matrix ordered by Dendrogram
    y = hierarchy.linkage(similarity_matrix, method='centroid')
    z = hierarchy.dendrogram(y, orientation='right', no_plot=True)
    index = z['leaves']
    similarity_matrix = similarity_matrix[index, :]
    similarity_matrix = similarity_matrix[:, index]
    ax = sns.heatmap(similarity_matrix, xticklabels=False, yticklabels=False)
    ax.set_title(f'Pairwise Similarity for n_clusters={n_clusters} (ordered)')
    plt.savefig(out_figure2, format=figure_format)


def group_similarity(participants: str, labels_files: list, metric: str, out_table1: str, out_table2: str,
                     out_figure1: str, out_figure2: str, out_figure3: str, figure_format: str = 'png'):
    """Group Similarity (subject similarity to group clustering)

    Parameters
    ----------
    participants : str
        Path to the participant information dataframe file (as .tsv)
    labels_files : list
        List of file paths to the various group clustering files.
    metric : str, {'adjusted rand index', 'v measure', 'adjusted mutual information'}
        Name of the similarity metric used to generate the similarity scores
    out_table1 : str
        Output file path for table 1
    out_table2 : str
        Output file path for table 2
    out_figure1 : str
        Output file path for figure 1
    out_figure2 : str
        Output file path for figure 2
    out_figure3 : str
        Output file path for figure 3
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

    data.to_csv(out_table1, sep='\t', index=False)
    reference_data.to_csv(out_table2, sep='\t', index=False)
    plt.ioff()

    # Generate Similarity Figure
    ax = sns.boxplot(x='clusters', y='similarity', data=data, showfliers=False)
    ax.set_title('Similarity of individual- to group-clusters')
    sns.despine(offset=10, trim=True)
    plt.savefig(out_figure1, format=figure_format)
    plt.clf()

    # Generate Relabeling Accuracy Figure
    ax = sns.boxplot(x='clusters', y='relabel accuracy', data=data, showfliers=False)
    ax.set_title('Relabeling accuracy of individual- to group-level reference')
    sns.despine(offset=10, trim=True)
    plt.savefig(out_figure2, format=figure_format)
    plt.clf()

    # Generate Cophenetic Correlation Figure
    ax = sns.pointplot(x='clusters', y='cophenetic correlation', data=reference_data)
    ax.set_title('Cophenetic Correlation of Group-level Clustering')
    sns.despine(offset=10, trim=True)
    plt.savefig(out_figure3, format=figure_format)
