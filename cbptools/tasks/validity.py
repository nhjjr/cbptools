from cbptools.cluster import davies_bouldin_score, \
    weak_deletion_stability_score
from cbptools.utils import sort_files
from sklearn.metrics import silhouette_score, calinski_harabasz_score, \
    adjusted_rand_score, v_measure_score, adjusted_mutual_info_score
import pandas as pd
import itertools
import numpy as np
import os


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

    _, ext = os.path.splitext(connectivity)
    connectivity = np.load(connectivity)

    if ext == '.npz':
        connectivity = connectivity.get('connectivity')

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


def merge_internal_validity(participants: str, validity: list, metrics: list,
                            out: str) -> None:
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
    out : str
        Output file path for tabular results
    """

    if not metrics:
        raise ValueError('Internal validity metrics must be set')

    # Merge internal validity scores by metric into subject x k
    # (n_clusters) tables
    validity = sort_files(participants, validity, pos=-1, sep='_',
                          index_col='participant_id')
    data = pd.concat((pd.read_csv(f, sep='\t', index_col=False)
                      for f in validity), ignore_index=True)
    data.to_csv(out, sep='\t', index=False)


def similarity(participants: str, labels_files: str, metric: str, out1: str,
               out2: str, out3: str) -> None:
    """Pairwise similarity matrix between all participant clustering results

    Parameters
    ----------
    participants : str
        Path to the participant information dataframe file (as .tsv)
    labels_files : list
        List of file paths to the various group clustering files.
    metric : str, {'adjusted rand index', 'v measure', 'adjusted
        mutual information'}. Name of the similarity metric used to
        generate the similarity scores
    out1 : str
        Output file path for individual similarity matrices
    out2 : str
        Output file path for group similarity scores
    out3 : str
        Output file path for cophenetic correlation
    """

    metric = metric.lower()
    if metric == 'adjusted_rand_score':
        metric = adjusted_rand_score

    elif metric == 'v_measure_score':
        metric = v_measure_score

    elif metric == 'adjusted_mutual_info_score':
        metric = adjusted_mutual_info_score

    else:
        raise ValueError('Metric \'{metric}\' not recognized')

    participants = pd.read_csv(participants, sep='\t')['participant_id']
    df = pd.DataFrame(columns=['participant_id', 'clusters', 'similarity',
                               'relabel accuracy'])
    df_reference = pd.DataFrame(columns=['clusters', 'cophenetic correlation'])
    similarity_matrices = dict()

    for file in labels_files:
        data = np.load(file)
        individual_labels = data.get('individual_labels')
        relabel_accuracy = data.get('relabel_accuracy')
        group_labels = data.get('group_labels')
        n_clusters = len(set(group_labels))

        # Individual Similarity
        n_participants = individual_labels.shape[0]
        similarity_matrix = np.zeros((n_participants, n_participants))

        for (a_index, a), (b_index, b) in \
                itertools.combinations(enumerate(individual_labels), 2):
            similarity_matrix[a_index, b_index] = metric(a, b)

        similarity_matrix += similarity_matrix.T
        np.fill_diagonal(similarity_matrix, 1)
        name = 'individual_similarity_%sclusters' % n_clusters
        similarity_matrices[name] = similarity_matrix

        # Group Similarity
        df_reference = df_reference.append({
            'clusters': n_clusters,
            'cophenetic correlation': data.get('cophenetic_correlation')
        }, ignore_index=True)

        for participant_id, labels, accuracy \
                in zip(participants, individual_labels, relabel_accuracy):
            df = df.append({
                'participant_id': str(participant_id),
                'clusters': n_clusters,
                'similarity': metric(group_labels, labels),
                'relabel accuracy': accuracy
            }, ignore_index=True)

    np.savez(out1, **similarity_matrices)
    df_reference.clusters = df_reference.clusters.astype(int)
    df.clusters = df.clusters.astype(int)
    df.to_csv(out2, sep='\t', index=False)
    df_reference.to_csv(out3, sep='\t', index=False)
