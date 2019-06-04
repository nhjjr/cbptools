import pandas as pd


def get_participant_ids(file: pd.DataFrame, sep: str = None,
                        index_col: str = 'participant_id') -> list:
    participant_ids = pd.read_csv(
        file,
        sep=sep,
        engine='python'
    ).get(index_col).tolist()

    return participant_ids


def expected_output(n_clusters: list, figure_format: str = 'png',
                    internal_validity_metrics: list = None):
    out = []

    for k in n_clusters:
        files = [
            'clustering/clustering_group_k%s.npz' % k,
            'summary/niftis/group_clustering_k%s.nii' % k,
            'summary/individual_similarity_%s_clusters.npy' % k,
            'summary/figures/individual_similarity_%sclusters_heatmap.%s'
            % (k, figure_format),
            'summary/figures/individual_similarity_%sclusters_clustermap.%s'
            % (k, figure_format)
        ]
        out += files

    out += [
        'summary/group_similarity.tsv',
        'summary/cophenetic_correlation.tsv',
        'summary/figures/group_scores.%s' % figure_format
    ]

    if internal_validity_metrics:
        out += [
            'summary/internal_validity.tsv',
            'summary/figures/internal_validity.%s' % figure_format
        ]

    return sorted(out)
