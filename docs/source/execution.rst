Execution
=========
Once the project has been successfully created, change the directory to the project directory (`workdir`). This is the
directory the `Snakefile` is located in. From there, a snakemake command can be used in the terminal to start
processing the data.

For a simple local run using 8 threads and at most 20GB of memory, run:

.. code-block:: bash

    snakemake -j 8 --resources mem_mb=20000


To execute the pipeline on a cluster using SLURM, make sure to first modify the cluster.json file so that all SLURM
parameters are set properly.

.. code-block:: bash

    snakemake -j 999 -w 240 -u cluster.json --resources -c "sbatch -p {cluster.partition} -n {cluster.n} -N {cluster.N} -t {cluster.time} -c {cluster.c} --mem-per-cpu={cluster.mem} --out={cluster.out} --job-name={cluster.name}"

Here, `-c "sbatch ..."` defines the command snakemake uses to start jobs through SLURM's sbatch. The `-w 240` is added
to ensure that snakemake waits at least 240 seconds for files to appear on the file system. File system latency may
cause snakemake to assume something went wrong with generating the file.

File system latency may cause issues with particularly large input files. For instance, the `connectivity` task uses
time-series images as input, which can easily be as large as 5 GB. When all parallel job executions of this task
try to simultaneously load such big files, the jobs will slow one another down. For this purpose, the `io` resource has
been added to the 'rsfmri' and 'dmri' connectivity and probtrackx2 tasks. Simply add `io=x` to the `--resources`
parameter, where x is the number of simultaneous jobs you think your file system can manage. For example:

.. code-block:: bash

    snakemake -j 999 -w 240 -u cluster.json --resources io=10 -c "sbatch -p {cluster.partition} -n {cluster.n} -N {cluster.N} -t {cluster.time} -c {cluster.c} --mem-per-cpu={cluster.mem} --out={cluster.out} --job-name={cluster.name}"

Note that the timing of each task depends crucially on the current state of the cluster system and is therefore
difficult to predict. If you notice that jobs fail due to timeouts, the `time` parameter in `cluster.json` should be
increased.

The snakemake tool is very flexible and for more information on how to use it, read the
`snakemake documentation <https://snakemake.readthedocs.io/en/stable/index.html>`_.
