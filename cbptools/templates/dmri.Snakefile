
rule probtrackx2:
    input:
        seed = 'highres_seed_mask.nii',
        target = 'target_mask.nii',
        <cbptools['input_data:bet_binary_mask']>,
        <cbptools['input_data:xfm']>,
        <cbptools['input_data:inv_xfm']>
    output: 'probtrackx2/{participant_id}/fdt_matrix2.dot'
    threads: 1
    resources:
        mem_mb = <cbptools['!mem_mb:connectivity']>,
        io = 1
    params:
        outdir = 'probtrackx2/{participant_id}',
        <cbptools['input_data:samples']>,
        <cbptools['parameters:connectivity:dist_thresh']>,
        <cbptools['parameters:connectivity:loop_check']>,
        <cbptools['parameters:connectivity:c_thresh']>,
        <cbptools['parameters:connectivity:step_length']>,
        <cbptools['parameters:connectivity:n_samples']>,
        <cbptools['parameters:connectivity:n_steps']>,
        <cbptools['parameters:connectivity:correct_path_distribution']>
    shell:
        "probtrackx2 --seed={input.seed} --target2={input.target} --samples={params.samples} \
        --mask={input.bet_binary_mask} --xfm={input.xfm} --invxfm={input.inv_xfm} --dir={params.outdir} \
        --nsamples={params.n_samples} --nsteps={params.n_steps} --steplength={params.step_length} \
        --distthresh={params.dist_thresh} --cthr={params.c_thresh} --omatrix2 --forcedir {params.loop_check} \
        {params.correct_path_distribution} > /dev/null"


rule connectivity:
    input:
        fdt_matrix2 = 'probtrackx2/{participant_id}/fdt_matrix2.dot',
        seed = 'seed_mask.nii'
    output: 'connectivity/connectivity_{participant_id}.npy'
    resources:
        mem_mb = <cbptools['!mem_mb:clustering']>
    params:
        <cbptools['parameters:connectivity:cleanup_fsl']>,
        <cbptools['parameters:connectivity:pca_transform']>,
        <cbptools['parameters:connectivity:cubic_transform']>
    run:
        tasks.connectivity_dmri(
            fdt_matrix2=input.fdt_matrix2, seed=input.seed, out=output[0], cleanup_fsl=params.cleanup_fsl,
            pca_transform=params.pca_transform, cubic_transform=params.cubic_transform
        )
