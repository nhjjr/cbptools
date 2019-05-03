#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cbptools.cluster import davies_bouldin_score, find_centers, weak_deletion_stability, gap_score
from sklearn.metrics import silhouette_score, calinski_harabaz_score
import numpy as np
import pandas as pd


def main(connectivity: str, labels: list, participant_id: str, metrics: list, file: str):
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
    file : str
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

    df.to_csv(file, sep='\t', index=False)


if __name__ == '__main__':
    main(
        connectivity=snakemake.input.get('connectivity'),
        labels=snakemake.input.get('labels'),
        participant_id=snakemake.wildcards.get('participant_id'),
        metrics=snakemake.params.get('metrics'),
        file=snakemake.output[0]
    )
