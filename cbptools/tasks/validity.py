from cbptools.utils import sort_files
from cbptools import cluster

from sklearn import metrics
import nibabel as nib
import pandas as pd
import itertools
import numpy as np
import os


def internal_validity(input: dict, output: dict, params: dict) -> None:
    """ Give single-participant connectivity and associated cluster
    label files for each requested number of clusters as input and
    this script returns the requested validity indices for each set
    of labels.

    Parameters
    ----------
    input : dict
        Input files, allowed: {connectivity, labels}
    output : dict
        Output file, allowed {scores}
    params : dict
        Parameters, allowed {participant_id, metrics}. The options parameter 
        is equivalent to validity:internal in the CBPtools documentation on 
        readthedocs.io under the parameters for 'clustering'.
    """

    # input, output, params
    connectivity_file = input.get('connectivity')
    labels_files = input.get('labels')
    scores_file = output.get('scores')
    participant_id = params.get('participant_id')
    validity_metrics = params.get('metrics')

    columns = ['participant_id', 'n_clusters'] + validity_metrics
    df = pd.DataFrame(columns=columns)
    _, ext = os.path.splitext(connectivity_file)
    connectivity = np.load(connectivity_file)

    if ext == '.npz':
        connectivity = connectivity.get('connectivity')

    for label in labels_files:
        label = np.load(label) + 1
        df = df.append({
            'participant_id': participant_id,
            'n_clusters': len(set(label))
        }, ignore_index=True)
        idx = df.iloc[-1].name

        for metric in validity_metrics:
            if hasattr(metrics, metric):
                f = getattr(metrics, metric)
            elif hasattr(cluster, metric):
                f = getattr(cluster, metric)
            else:
                raise ValueError('Validity metric %s not found' % metric)

            score = f(connectivity, label)
            df.loc[idx, metric] = score

    df.to_csv(scores_file, sep='\t', index=False)


def merge_internal_validity(input: dict, output: dict) -> None:
    """ Generate a summary of the internal validity results.

    This script merges internal cluster validity into one table and
    generates a figure for a summary viewing.

    Parameters
    ----------
    input : dict
        Input files, allowed: {participants, validity}
    output : dict
        Output file, allowed {scores}
    """

    participants_file = input.get('participants')
    validity_files = input.get('validity')
    scores_file = output.get('scores')

    # Merge internal validity scores by metric into subject x k tables
    validity = sort_files(participants_file, validity_files, pos=1, sep='/',
                          index_col='participant_id')
    data = pd.concat((pd.read_csv(f, sep='\t', index_col=False)
                      for f in validity), ignore_index=True)
    data.to_csv(scores_file, sep='\t', index=False)


def individual_similarity(input: dict, output: dict, params: dict) -> None:
    """Pairwise similarity matrix between all participant clustering results

    Parameters
    ----------
    input : dict
        Input files, allowed: {labels}
    output : dict
        Output file, allowed {individual_similarity}
    params : dict
        Parameters, allowed {metric}. The options parameter is equivalent to
        validity:similarity in the CBPtools documentation on readthedocs.io
        under the parameters for 'clustering'.
    """

    labels_file = input.get('labels')
    similarity_file = output.get('individual_similarity_matrix')
    metric = params.get('metric').lower()
    data = np.load(labels_file)
    individual_labels = data.get('individual_labels')

    if hasattr(metrics, metric):
        f = getattr(metrics, metric)
    else:
        raise ValueError('Metric %s not recognized' % metric)

    # Individual Similarity
    n_subjects = individual_labels.shape[0]
    similarity_matrix = np.zeros((n_subjects, n_subjects))

    for (a_index, a), (b_index, b) in \
            itertools.combinations(enumerate(individual_labels), 2):
        similarity_matrix[a_index, b_index] = f(a, b)

    similarity_matrix += similarity_matrix.T
    np.fill_diagonal(similarity_matrix, 1)

    # Save similarity matrix to disk
    np.save(similarity_file, similarity_matrix)


def group_similarity(input: dict, output: dict, params: dict) -> None:
    """Pairwise similarity matrix between all participant clustering results

    Parameters
    ----------
    input : dict
        Input files, allowed: {participants, labels}
    output : dict
        Output file, allowed {group_similarity, cophenetic_correlation}
    params : dict
        Parameters, allowed {metric}. The options parameter is equivalent to
        validity:similarity in the CBPtools documentation on readthedocs.io
        under the parameters for 'clustering'.
    """
    # input, output, params
    participants_file = input.get('participants')
    labels_files = input.get('labels')
    similarity_file = output.get('group_similarity')
    cophenet_file = output.get('cophenetic_correlation')
    metric = params.get('metric').lower()

    participants = pd.read_csv(participants_file, sep='\t')
    participants = participants['participant_id']
    df = pd.DataFrame(columns=['participant_id', 'clusters', 'similarity',
                               'relabel accuracy'])
    df_reference = pd.DataFrame(columns=['clusters', 'cophenetic correlation'])

    if hasattr(metrics, metric):
        f = getattr(metrics, metric)
    else:
        raise ValueError('Metric %s not recognized' % metric)

    for file in labels_files:
        data = np.load(file)
        ilabels = data.get('individual_labels')
        glabels = data.get('group_labels')
        accuracy = data.get('relabel_accuracy')

        # Obtain cluster number from file name
        n_clusters = file.split('/')
        n_clusters = [i for i in n_clusters if 'clusters' in i][0]
        n_clusters = n_clusters.replace('clusters', '')

        if n_clusters.isdigit():
            n_clusters = int(n_clusters)
        else:
            raise ValueError('Could not derive cluster number from file name')

        # Group similarity & cophenetic correlation
        df_reference = df_reference.append({
            'clusters': n_clusters,
            'cophenetic correlation': data.get('cophenetic_correlation')
        }, ignore_index=True)

        for ppid, labels, acc in zip(participants, ilabels, accuracy):
            df = df.append({
                'participant_id': str(ppid),
                'clusters': n_clusters,
                'similarity': f(glabels, labels),
                'relabel accuracy': acc
            }, ignore_index=True)

    df_reference.clusters = df_reference.clusters.astype(int)
    df.clusters = df.clusters.astype(int)
    df.to_csv(similarity_file, sep='\t', index=False)
    df_reference.to_csv(cophenet_file, sep='\t', index=False)


def reference_similarity(input: dict, output: dict, params: dict) -> None:
    """Pairwise similarity matrix between all references and all group
    clustering results.

    Parameters
    ----------
    input : dict
        Input files, allowed: {references, labels}
    output : dict
        Output file, allowed {figure, scores}
    params : dict
        Parameters, allowed {metric}.
    """

    reference_files = input.get('references')
    label_files = input.get('labels')
    similarity_file = output.get('similarity')
    metric = params.get('metric').lower()
    n_clusters = params.get('n_clusters')

    if hasattr(metrics, metric):
        f = getattr(metrics, metric)
    else:
        raise ValueError('Metric %s not recognized' % metric)

    columns = ['k=%s' % k for k in n_clusters]
    columns += ['reference']
    df = pd.DataFrame(columns=columns)

    for reference_file in reference_files:
        reference = nib.load(reference_file)
        reference = reference.get_data()
        reference = reference[np.where(reference > 0)]
        scores = dict()

        for label_file in label_files:
            labels = np.load(label_file)
            labels = labels.get('group_labels')
            labels += 1  # prevent 0-indexing
            n_clusters = np.unique(labels).size
            scores['k=%s' % n_clusters] = f(reference, labels)

        df = df.append({
            'reference': os.path.basename(reference_file),
            **scores
        }, ignore_index=True)

    df.set_index('reference', inplace=True)
    df.to_csv(similarity_file, sep='\t')
