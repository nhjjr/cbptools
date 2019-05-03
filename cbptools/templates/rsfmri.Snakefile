
rule connectivity:
    input:
        seed = 'seed_mask.nii',
        target = 'target_mask.nii',
        time_series = <cbptools['input_data:time_series']>
    output: 'connectivity/connectivity_{participant_id}.npy'
    threads: 1
    resources: mem_mb=10000, io=5
    params:
        confounds = <cbptools['input_data:confounds']>,
        arctanh_transform = <cbptools['parameters:connectivity:arctanh_transform']>,
        pca_transform = <cbptools['parameters:connectivity:pca_transform']>,
        high_pass = <cbptools['parameters:connectivity:high_pass']>,
        low_pass = <cbptools['parameters:connectivity:low_pass']>,
        tr = <cbptools['parameters:connectivity:tr']>,
        smoothing_fwhm = <cbptools['parameters:connectivity:smoothing_fwhm']>,
        seed_low_variance_threshold = <cbptools['parameters:connectivity:seed_low_variance_threshold']>,
        target_low_variance_threshold = <cbptools['parameters:connectivity:target_low_variance_threshold']>,
        log = '%s/connectivity/connectivity_{participant_id}.tsv' % os.getcwd()
    run:
        tasks.connectivity_fmri(
            time_series=input.time_series, seed=input.seed, target=input.target,
            participant_id=wildcards.participant_id, file=output[0], log_file=params.log, confounds=params.confounds,
            arctanh_transform=params.arctanh_transform, pca_transform=params.pca_transform, high_pass=params.high_pass,
            low_pass=params.low_pass, tr=params.tr, smoothing_fwhm=params.smoothing_fwhm,
            seed_low_variance_threshold=params.seed_low_variance_threshold,
            target_low_variance_threshold=params.target_low_variance_threshold
        )

