
rule connectivity:
    input:
        seed = 'highres_seed_mask.nii',
        target = 'target_mask.nii',
        <cbptools['input_data:bet_binary_mask']>,
        <cbptools['input_data:xfm']>,
        <cbptools['input_data:inv_xfm']>
    output: 'connectivity/connectivity_{participant_id}.npy'
    threads: 1
    resources: mem_mb=10000, io=5
    params:
        <cbptools['input_data:samples']>,
        <cbptools['parameters:connectivity:pca_transform']>,
        <cbptools['parameters:connectivity:dist_thresh']>,
        <cbptools['parameters:connectivity:loop_check']>,
        <cbptools['parameters:connectivity:c_thresh']>,
        <cbptools['parameters:connectivity:step_length']>,
        <cbptools['parameters:connectivity:n_samples']>,
        <cbptools['parameters:connectivity:n_steps']>,
        <cbptools['parameters:connectivity:correct_path_distribution']>,
        <cbptools['parameters:connectivity:wait_for_file']>,
        <cbptools['parameters:connectivity:cleanup_fsl']>,
        <cbptools['parameters:connectivity:cubic_transform']>,
        tmp_dir = '%s/probtrackx2_{participant_id}' % os.getcwd()
    run:
        tasks.connectivity_dmri(
            seed=input.seed, target=input.target, samples=params.samples, bet_binary_mask=input.bet_binary_mask,
            tmp_dir=params.tmp_dir, xfm=input.xfm, inv_xfm=input.inv_xfm, out=output[0],
            pca_transform=params.pca_transform, dist_thresh=params.dist_thresh, loop_check=params.loop_check,
            c_thresh=params.c_thresh, step_length=params.step_length, n_samples=params.n_samples,
            n_steps=params.n_steps, pd=params.correct_path_distribution, wait_for_file=params.wait_for_file,
            cleanup_fsl=params.cleanup_fsl, cubic_transform=params.cubic_transform
        )

