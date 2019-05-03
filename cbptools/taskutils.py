import pandas as pd


def get_participant_ids(file: pd.DataFrame, sep: str = None, index_col: str = 'participant_id') -> list:
    participant_ids = pd.read_csv(file, sep=sep, engine='python').get(index_col).tolist()

    return participant_ids


def expected_output(n_clusters: list, ext: str = 'png', metrics: list = None):
    out = []

    for k in n_clusters:
        files = [
            f'clustering/clustering_group_k{k}.npz',
            f'summary/niftis/group_clustering_k{k}.nii',
            f'summary/individual_similarity_{k}_clusters.npy',
            f'summary/figures/individual_similarity_{k}clusters_unordered.{ext}',
            f'summary/figures/individual_similarity_{k}clusters_ordered.{ext}'
        ]
        out += files

    out += [
        'summary/group_similarity.tsv',
        'summary/cophenetic_correlation.tsv',
        f'summary/figures/group_similarity.{ext}',
        f'summary/figures/relabel_accuracy.{ext}',
        f'summary/figures/cophenetic_correlation.{ext}'
    ]

    if metrics:
        out += [
            'summary/internal_validity.tsv',
            'summary/figures/internal_validity.{ext}'
        ]

    return sorted(out)
