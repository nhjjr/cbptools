
rule connectivity:
    input:
        seed = 'seed_mask.nii',
        target = 'target_mask.nii',
        <cbptools['input_data:time_series']>,
        <cbptools['input_data:confounds']>
    output:
        connectivity_matrix = 'connectivity/connectivity_{participant_id}.npy'
    threads: 1
    resources:
        mem_mb = <cbptools['!mem_mb:connectivity']>,
        io = 1
    params:
        <cbptools['+input_data:confounds:sep']>,
        <cbptools['+input_data:confounds:usecols']>,
        <cbptools['parameters:connectivity:arctanh_transform']>,
        <cbptools['parameters:connectivity:pca_transform']>,
        <cbptools['parameters:connectivity:high_pass']>,
        <cbptools['parameters:connectivity:low_pass']>,
        <cbptools['parameters:connectivity:tr']>,
        <cbptools['parameters:connectivity:smoothing_fwhm']>,
        <cbptools['parameters:connectivity:seed_low_variance_threshold']>,
        <cbptools['parameters:connectivity:target_low_variance_threshold']>,
        log_file = '%s/connectivity/connectivity_{participant_id}.tsv' % os.getcwd()
    run:
        tasks.connectivity_fmri(
            time_series=input.time_series, seed=input.seed, target=input.target, confounds=input.get('confounds', None),
            participant_id=wildcards.participant_id, out=output.connectivity_matrix, log_file=params.log_file,
            sep=params.sep, usecols=params.usecols, arctanh_transform=params.arctanh_transform,
            pca_transform=params.pca_transform, band_pass=(params.high_pass, params.low_pass, params.tr),
            smoothing_fwhm=params.smoothing_fwhm, seed_low_variance=params.seed_low_variance_threshold,
            target_low_variance=params.target_low_variance_threshold
        )


rule merge_connectivity_logs:
    input:
        labels = expand(
            'clustering/clustering_k{n_clusters}_{participant_id}.npy',
            n_clusters=n_clusters,
            participant_id=participants
        ),
        connectivity_matrices = expand('connectivity/connectivity_{participant_id}.npy', participant_id=participants)
    output: 'log/connectivity_log.tsv'
    params:
        log_file = lambda wildcards: '%s/connectivity/connectivity_{participant_id}.tsv' % os.getcwd()
    threads: 1
    run:
        tasks.merge_connectivity_logs(log_file=params.log_file, participants=participants, out=output[0])


rule validate_connectivity:
    input:
        labels = expand(
            'clustering/clustering_k{n_clusters}_{participant_id}.npy',
            n_clusters=n_clusters,
            participant_id=participants
        ),
        log_file = 'log/connectivity_log.tsv'
    output:
        <cbptools['input_data:touchfile']>
    params:
        connectivity_matrix = lambda wildcards: 'connectivity/connectivity_{participant_id}.npy',
        cluster_labels = lambda wildcards: 'clustering/clustering_k{n_clusters}_{participant_id}.npy',
        <cbptools['parameters:clustering:n_clusters']>
    threads: 1
    run:
        tasks.validate_connectivity(
            log_file=input.log_file, connectivity=params.connectivity_matrix, labels=params.cluster_labels,
            n_clusters=params.n_clusters, out=output.touchfile
        )

