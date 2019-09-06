
rule probtrackx2:
    input:
        seed = 'highres_seed_mask.nii.gz',
        target = 'target_mask.nii.gz',
        <cbptools['data:bet_binary_mask']>,
        <cbptools['data:xfm']>,
        <cbptools['data:inv_xfm']>
    output: 'probtrackx2/{participant_id}/fdt_matrix2.dot'
    threads: 1
    resources:
        mem_mb = <cbptools['!mem_mb:connectivity']>,
        io = 1
    params:
        outdir = 'probtrackx2/{participant_id}',
        <cbptools['data:samples']>,
        <cbptools['parameters:probtract_proc:dist_thresh']>,
        <cbptools['parameters:probtract_proc:loop_check']>,
        <cbptools['parameters:probtract_proc:c_thresh']>,
        <cbptools['parameters:probtract_proc:step_length']>,
        <cbptools['parameters:probtract_proc:n_samples']>,
        <cbptools['parameters:probtract_proc:n_steps']>,
        <cbptools['parameters:probtract_proc:correct_path_distribution']>
    shell:
        "probtrackx2 --seed={input.seed} --target2={input.target} --samples={params.samples} \
        --mask={input.bet_binary_mask} --xfm={input.xfm} --invxfm={input.inv_xfm} --dir={params.outdir} \
        --nsamples={params.n_samples} --nsteps={params.n_steps} --steplength={params.step_length} \
        --distthresh={params.dist_thresh} --cthr={params.c_thresh} --omatrix2 --forcedir {params.loop_check} \
        {params.correct_path_distribution} > /dev/null"


rule connectivity:
    input:
        fdt_matrix2 = 'probtrackx2/{participant_id}/fdt_matrix2.dot',
        seed = 'seed_mask.nii.gz'
    output: 'connectivity/connectivity_{participant_id}.npz'
    resources:
        mem_mb = <cbptools['!mem_mb:clustering']>
    params:
        apply_cubic_transform = <cbptools['!parameters:probtract_proc:cubic_transform:apply']>,
        apply_pca = <cbptools['!parameters:probtract_proc:pca_transform:apply']>,
        pca_components = <cbptools['!parameters:probtract_proc:pca_transform:components']>,
        <cbptools['parameters:probtract_proc:compress']>,
        <cbptools['parameters:probtract_proc:cleanup_fsl']>,
    run:
        tasks.connectivity_dmri(
            fdt_matrix2=input.fdt_matrix2, seed=input.seed, out=output[0], cleanup_fsl=params.cleanup_fsl,
            apply_pca=params.apply_pca, pca_components=params.pca_components,
            apply_cubic_transform=params.apply_cubic_transform, compress_output=params.compress
        )
