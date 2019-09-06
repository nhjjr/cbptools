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
                    internal: list = None) -> list:
    out = []

    # Add cluster labels
    out += ['clustering/clustering_group_k%s.npz' % k for k in n_clusters]

    # Add NIfTI images with cluster labels
    out += ['summary/niftis/group_clustering_k%s.nii' % k for k in n_clusters]

    # Add individual similarity scores
    out += ['summary/individual_similarity.npz']

    # Add Heat- and clustermaps
    out += ['summary/figures/individual_similarity_%sclusters_heatmap.png'
            % k for k in n_clusters]
    out += ['summary/figures/individual_similarity_%sclusters_clustermap.png'
            % k for k in n_clusters]

    # Add group similarity, cophenetic correlation, and group scores
    out += ['summary/group_similarity.tsv',
            'summary/cophenetic_correlation.tsv']
    out += ['summary/figures/group_similarity.%s' % figure_format,
            'summary/figures/relabeling_accuracy.%s' % figure_format,
            'summary/figures/cophenetic_correlation.%s' % figure_format]

    # Add internal validity metrics
    if internal:
        out += ['summary/internal_validity.tsv']
        out += ['summary/figures/internal_validity_%s.%s'
                % (metric, figure_format)
                for metric in internal]

    # Add 3D volumetric plots
    out += ['summary/figures/group_clustering_k%s_%s.%s'
            % (k, view, figure_format) for k in n_clusters for view in
            ['right', 'left', 'superior', 'inferior', 'posterior', 'anterior']]

    return sorted(out)
