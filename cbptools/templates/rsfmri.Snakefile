
rule connectivity:
    input:
        seed = 'seed_mask.nii.gz',
        target = 'target_mask.nii.gz',
        <cbptools['data:time_series']>,
        <cbptools['data:confounds:file']>
    output:
        connectivity = 'connectivity/connectivity_{participant_id}.npz',
        log_file = temp('connectivity/connectivity_{participant_id}.tsv')
    threads: 1
    resources:
        mem_mb = <cbptools['!mem_mb:connectivity']>,
        io = 1
    params:
        delimiter = <cbptools['!data:confounds:delimiter']>,
        columns = <cbptools['!data:confounds:columns']>,
        apply_arctanh = <cbptools['!parameters:time_series_proc:arctanh_transform:apply']>,
        apply_pca = <cbptools['!parameters:time_series_proc:pca_transform:apply']>,
        pca_components = <cbptools['!parameters:time_series_proc:pca_transform:components']>,
        apply_bandpass = <cbptools['!parameters:time_series_proc:band_pass_filtering:apply']>,
        bandpass_band = <cbptools['!parameters:time_series_proc:band_pass_filtering:band']>,
        bandpass_tr = <cbptools['!parameters:time_series_proc:band_pass_filtering:tr']>,
        apply_smoothing = <cbptools['!parameters:time_series_proc:smoothing:apply']>,
        smoothing_fwhm = <cbptools['!parameters:time_series_proc:smoothing:fwhm']>,
        apply_low_variance_threshold = <cbptools['!parameters:time_series_proc:low_variance_error:apply']>,
        low_variance_in_seed = <cbptools['!parameters:time_series_proc:low_variance_error:in_seed']>,
        low_variance_in_target = <cbptools['!parameters:time_series_proc:low_variance_error:in_target']>,
        low_variance_behavior = <cbptools['!parameters:time_series_proc:low_variance_error:behavior']>,
        <cbptools['parameters:time_series_proc:compress']>
    run:
        tasks.connectivity_fmri(
            time_series=input.time_series, seed=input.seed, target=input.target, confounds=input.get('confounds', None),
            log_file=output.log_file, participant_id=wildcards.participant_id, out=output.connectivity,
            sep=params.delimiter, usecols=params.columns, apply_arctanh=params.apply_arctanh,
            apply_pca=params.apply_pca, pca_components=params.pca_components, apply_bandpass=params.apply_bandpass,
            bandpass_band=params.bandpass_band, bandpass_tr=params.bandpass_tr, apply_smoothing=params.apply_smoothing,
            smoothing_fwhm=params.smoothing_fwhm, apply_low_variance_threshold=params.apply_low_variance_threshold,
            low_variance_in_seed=params.low_variance_in_seed, low_variance_in_target=params.low_variance_in_target,
            low_variance_behavior=params.low_variance_behavior, compress_output=params.compress
        )


rule merge_connectivity_logs:
    input: expand('connectivity/connectivity_{participant_id}.tsv', participant_id=participants)
    output: 'log/connectivity_log.tsv'
    threads: 1
    run:
        tasks.merge_connectivity_logs(log_files=input, out=output[0])


rule validate_connectivity:
    input:
        labels = expand(
            'clustering/clustering_k{n_clusters}_{participant_id}.npy',
            n_clusters=n_clusters,
            participant_id=participants
        ),
        log_file = 'log/connectivity_log.tsv'
    output:
        <cbptools['data:touchfile']>
    params:
        connectivity = lambda wildcards: 'connectivity/connectivity_{participant_id}.npy',
        cluster_labels = lambda wildcards: 'clustering/clustering_k{n_clusters}_{participant_id}.npy',
        <cbptools['parameters:clustering:n_clusters']>
    threads: 1
    run:
        tasks.validate_connectivity(
            log_file=input.log_file, connectivity=params.connectivity, labels=params.cluster_labels,
            n_clusters=params.n_clusters, out=output.touchfile
        )

