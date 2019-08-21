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

    # Add cluster labels
    out += ['clustering/clustering_group_k%s.npz' % k for k in n_clusters]

    # Add NIfTI images with cluster labels
    out += ['summary/niftis/group_clustering_k%s.nii' % k for k in n_clusters]

    # Add individual similarity scores
    out += ['summary/individual_similarity_%s_clusters.npy' % k
            for k in n_clusters]

    # Add Heat- and clustermaps
    out += ['summary/figures/individual_similarity_%sclusters_heatmap.%s'
            % (k, figure_format) for k in n_clusters]
    out += ['summary/figures/individual_similarity_%sclusters_clustermap.%s'
            % (k, figure_format) for k in n_clusters]

    # Add group similarity, cophenetic correlation, and group scores
    out += ['summary/group_similarity.tsv',
            'summary/cophenetic_correlation.tsv',
            'summary/figures/group_scores.%s' % figure_format]

    # Add internal validity metrics
    if internal_validity_metrics:
        out += ['summary/internal_validity.tsv',
                'summary/figures/internal_validity.%s' % figure_format]

    # Add 3D volumetric plots
    out += ['summary/figures/group_clustering_k%s_%s.%s'
            % (k, view, figure_format) for k in n_clusters for view in
            ['right', 'left', 'superior', 'inferior', 'posterior', 'anterior']]

    return sorted(out)
