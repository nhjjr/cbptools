
rule participant_level_clustering:
    input: <cbptools['input_data:connectivity_matrix']>
    output: 'clustering/clustering_k{k}_{participant_id}.npy'
    threads: 1
    resources: mem_mb=2000  # TODO: Change this later
    params:
        algorithm = <cbptools['parameters:clustering:algorithm']>,
        init = <cbptools['parameters:clustering:init']>,
        max_iter = <cbptools['parameters:clustering:max_iter'],
        n_init = <cbptools['parameters:clustering:n_init']>,
        n_clusters = lambda wildcards: int(wildcards.k)
    run:
        tasks.participant_level_clustering(
            connectivity=input[0], out=output[0], algorithm=params.algorithm, init=params.init,
            max_iter=params.max_iter, n_clusters=params.n_clusters, n_init=params.n_init
        )


rule group_level_clustering:
    input:
        seed_img = 'seed_mask.nii',
        participants = 'participants.tsv',
        labels = expand('clustering/clustering_k{k}_{participant_id}.npy', participant_id=participants, k='{k}')
    output:
        group_labels = 'clustering/clustering_group_k{k}.npz',
        group_img = 'summary/niftis/group_clustering_k{k}.nii'
    threads: 1
    resources: mem_mb=2000
    params:
        input_data_type = <cbptools['input_data_type']>,
        linkage = <cbptools['parameters:clustering:linkage']>,
        method = <cbptools['parameters:clustering:group_method']>
    run:
        tasks.group_level_clustering(
            seed_img=input.seed_img, participants=input.participants, individual_labels=input.labels,
            linkage=params.linkage, input_data_type=params.input_data_type, out_labels=output.group_labels,
            out_img=output.group_img, method=params.method
        )


rule internal_validity:
    input:
        connectivity = <cbptools['input_data:connectivity_matrix']>,
        labels = expand('clustering/clustering_k{k}_{participant_id}.npy', k=k, participant_id='{participant_id}')
    output: temp('validity/internal-validity_{participant_id}.tsv')
    threads: 1
    resources: mem_mb=2000
    params:
        metrics = <cbptools['parameters:clustering:internal_validity_metrics']>
    run:
        tasks.internal_validity(
            connectivity=input.connectivity, labels=input.labels, participant_id=wildcards.participant_id,
            metrics=params.metrics, out=output[0]
        )


rule summary_internal_validity:
    input:
        participants = 'participants.tsv',
        validity = expand('validity/internal-validity_{participant_id}.tsv', participant_id=participants, k=k)
    output:
        table = 'summary/internal_validity.tsv',
        figure = 'summary/figures/internal_validity.<cbptools['parameters:summary:figure_format']>'
    params:
        internal_validity_metrics = <cbptools['parameters:clustering:internal_validity_metrics']>,
        figure_format = <cbptools['parameters:summary:figure_format']>
    threads: 1
    run:
        tasks.summary_internal_validity(
            participants=input.participants, validity=input.validity,
            internal_validity_metrics=params.internal_validity_metrics, out_table=output.table,
            out_figure=output.figure, figure_format=params.figure_format
        )


rule individual_similarity:
    input:
        labels = 'clustering/clustering_group_k{k}.npz'
    output:
        matrix = 'summary/individual_similarity_{k}_clusters.npy',
        figure1 = 'summary/figures/individual_similarity_{{k}}clusters_unordered.<cbptools['parameters:summary:figure_format']>',
        figure2 = 'summary/figures/individual_similarity_{{k}}clusters_ordered.<cbptools['parameters:summary:figure_format']>'
    threads: 1
    params:
        metric = <cbptools['parameters:clustering:similarity_metric']>,
        figure_format = <cbptools['parameters:summary:figure_format']>
    run:
        tasks.individual_similarity(
            labels=input.labels, metric=params.metric, n_clusters=wildcards.k, out_matrix=output.matrix,
            out_figure1=output.figure1, out_figure2=output.figure2, figure_format=params.figure_format
        )


rule group_similarity:
    input:
        participants = 'participants.tsv',
        labels = expand('clustering/clustering_group_k{k}.npz', k=k)
    output:
        table1 = 'summary/group_similarity.tsv',
        table2 = 'summary/cophenetic_correlation.tsv',
        figure1 = 'summary/figures/group_similarity.<cbptools['parameters:summary:figure_format']>',
        figure2 = 'summary/figures/relabel_accuracy.<cbptools['parameters:summary:figure_format']>',
        figure3 = 'summary/figures/cophenetic_correlation.<cbptools['parameters:summary:figure_format']>'
    threads: 1
    params:
        metric = <cbptools['parameters:clustering:similarity_metric']>,
        figure_format = <cbptools['parameters:summary:figure_format']>
    run:
        tasks.group_similarity(
            participants=input.participants, labels_files=input.labels, metric=params.metric, out_table1=output.table1,
            out_table2=output.table2, out_figure1=output.figure1, out_figure2=output.figure2,
            out_figure3=output.figure3, figure_format=params.figure_format
        )
