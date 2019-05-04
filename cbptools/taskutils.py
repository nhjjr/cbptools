import pandas as pd


def get_participant_ids(file: pd.DataFrame, sep: str = None, index_col: str = 'participant_id') -> list:
    participant_ids = pd.read_csv(file, sep=sep, engine='python').get(index_col).tolist()

    return participant_ids


def expected_output(n_clusters: list, figure_format: str = 'png', internal_validity_metrics: list = None):
    out = []

    for k in n_clusters:
        files = [
            f'clustering/clustering_group_k{k}.npz',
            f'summary/niftis/group_clustering_k{k}.nii',
            f'summary/individual_similarity_{k}_clusters.npy',
            f'summary/figures/individual_similarity_{k}clusters_unordered.{figure_format}',
            f'summary/figures/individual_similarity_{k}clusters_ordered.{figure_format}'
        ]
        out += files

    out += [
        'summary/group_similarity.tsv',
        'summary/cophenetic_correlation.tsv',
        f'summary/figures/group_similarity.{figure_format}',
        f'summary/figures/relabel_accuracy.{figure_format}',
        f'summary/figures/cophenetic_correlation.{figure_format}'
    ]

    if internal_validity_metrics:
        out += [
            'summary/internal_validity.tsv',
            f'summary/figures/internal_validity.{figure_format}'
        ]

    return sorted(out)
