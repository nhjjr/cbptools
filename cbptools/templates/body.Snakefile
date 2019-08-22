
rule participant_level_clustering:
    input:
        <cbptools['input_data:connectivity_matrix']>
    output: temp('clustering/clustering_k{n_clusters}_{participant_id}.npy')
    threads: 1
    resources:
        mem_mb = <cbptools['!mem_mb:clustering']>
    params:
        <cbptools['parameters:clustering:algorithm']>,
        <cbptools['parameters:clustering:init']>,
        <cbptools['parameters:clustering:max_iter']>,
        <cbptools['parameters:clustering:n_init']>,
        n_clusters = lambda wildcards: int(wildcards.n_clusters)
    run:
        tasks.participant_level_clustering(
            connectivity=input.connectivity_matrix, out=output[0], algorithm=params.algorithm, init=params.init,
            max_iter=params.max_iter, n_clusters=params.n_clusters, n_init=params.n_init
        )


rule group_level_clustering:
    input:
        seed_img = 'seed_mask.nii',
        participants = 'participants.tsv',
        <cbptools['input_data:seed_indices']>,
        <cbptools['input_data:touchfile']>,
        labels = expand(
            'clustering/clustering_k{n_clusters}_{participant_id}.npy',
            participant_id=participants,
            n_clusters='{n_clusters}'
        )
    output:
        group_labels = 'clustering/clustering_group_k{n_clusters}.npz',
        group_img = 'summary/niftis/group_clustering_k{n_clusters}.nii'
    threads: 1
    resources:
        mem_mb = <cbptools['!mem_mb:clustering']>
    params:
        <cbptools['parameters:clustering:linkage']>,
        <cbptools['parameters:clustering:group_method']>
    run:
        tasks.group_level_clustering(
            seed_img=input.seed_img, participants=input.participants, individual_labels=input.labels,
            seed_indices=input.get('seed_indices', None), linkage=params.linkage, method=params.group_method,
            out_labels=output.group_labels, out_img=output.group_img
        )


rule internal_validity:
    input:
        <cbptools['input_data:connectivity_matrix']>,
        <cbptools['input_data:touchfile']>,
        labels = expand(
            'clustering/clustering_k{n_clusters}_{participant_id}.npy',
            n_clusters=n_clusters,
            participant_id='{participant_id}'
        )
    output: temp('validity/internal-validity_{participant_id}.tsv')
    threads: 1
    resources:
        mem_mb = <cbptools['!mem_mb:clustering']>
    params:
        <cbptools['parameters:clustering:internal_validity_metrics']>
    run:
        tasks.internal_validity(
            connectivity=input.connectivity_matrix, labels=input.labels, participant_id=wildcards.participant_id,
            metrics=params.internal_validity_metrics, out=output[0]
        )


rule merge_internal_validity:
    input:
        participants = 'participants.tsv',
        validity = expand(
            'validity/internal-validity_{participant_id}.tsv',
            participant_id=participants,
            n_clusters=n_clusters
        )
    output: 'summary/internal_validity.tsv'
    params:
        <cbptools['parameters:clustering:internal_validity_metrics']>
    threads: 1
    run:
        tasks.merge_internal_validity(
            participants=input.participants, validity=input.validity, metrics=params.internal_validity_metrics,
            out=output[0]
        )


rule similarity:
    input:
        participants = 'participants.tsv',
        labels = expand('clustering/clustering_group_k{n_clusters}.npz', n_clusters=n_clusters)
    output:
        individual_similarity = 'summary/individual_similarity.npz',
        group_similarity = 'summary/group_similarity.tsv',
        cophenetic_correlation = 'summary/cophenetic_correlation.tsv'
    threads: 1
    params:
        <cbptools['parameters:clustering:similarity_metric']>
    run:
        tasks.similarity(
            participants=input.participants, labels_files=input.labels, metric=params.similarity_metric,
            out1=output.individual_similarity, out2=output.group_similarity, out3=output.cophenetic_correlation
        )


rule plot_internal_validity:
    input: 'summary/internal_validity.tsv'
    output:
        figures = expand(
            'summary/figures/internal_validity_{metric}.<cbptools['!parameters:summary:figure_format']>',
            metric=<cbptools['!parameters:clustering:internal_validity_metrics']>  # TODO: Check if this returns a list
        )
    params:
        outdir='summary/figures',
        <cbptools['parameters:clustering:internal_validity_metrics']>,
        <cbptools['parameters:summary:figure_format']>
    threads: 1
    run:
        tasks.plot_internal_validity(
            internal_validity=input[0], metrics=params.internal_validity_metrics, outdir=params.outdir,
            figure_format=params.figure_format
        )


rule plot_similarity:
    input:
        individual_similarity = 'summary/individual_similarity.npz',
        group_similarity = 'summary/group_similarity.tsv',
        cophenetic_correlation = 'summary/cophenetic_correlation.tsv'
    output:
        expand('summary/figures/individual_similarity_{n_clusters}clusters_heatmap.png', n_clusters=n_clusters),
        expand('summary/figures/individual_similarity_{n_clusters}clusters_clustermap.png', n_clusters=n_clusters),
        'summary/figures/group_similarity.<cbptools['!parameters:summary:figure_format']>',
        'summary/figures/relabeling_accuracy.<cbptools['!parameters:summary:figure_format']>',
        'summary/figures/cophenetic_correlation.<cbptools['!parameters:summary:figure_format']>'
    threads: 1
    params:
        outdir='summary/figures',
        <cbptools['parameters:summary:figure_format']>
    run:
        tasks.plot_similarity(
            individual=input.individual_similarity, group=input.group_similarity, cophenet=input.cophenetic_correlation,
            outdir=params.outdir, figure_format=params.figure_format
        )


rule plot_labeled_roi:
    input:
        labels = expand('clustering/clustering_group_k{n_clusters}.npz', n_clusters=n_clusters),
        seed_img = 'seed_mask.nii'
    output:
        figures = expand(
            'summary/figures/group_clustering_k{n_clusters}_{view}.<cbptools['!parameters:summary:figure_format']>',
            n_clusters=n_clusters,
            view=['right', 'left', 'superior', 'inferior', 'posterior', 'anterior']
        )
    threads: 1
    params:
        outdir = 'summary/figures',
        <cbptools['parameters:summary:figure_format']>
    run:
        tasks.plot_labeled_roi(
            group_labels=input.labels, seed_img=input.seed_img, outdir=params.outdir,
            figure_format=params.figure_format
        )
