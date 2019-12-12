.. _execution:

=========
Execution
=========
Once the project has been successfully created, change the directory to the project directory (`workdir`). This is the
directory the `Snakefile` workflow is located in. From there, a snakemake command can be used in the terminal to start
processing the data. Snakemake is a very flexible command-line tool with many parameters to customize the execution of
the workflow. This page will list some common use-cases, but for a more in-depth description visit the
`Snakemake documentation <https://snakemake.readthedocs.io/en/stable/executable.html>`_.

Local execution
---------------
For a simple local run snakemake can be called directly without arguments. The below example shows a local run using
8 threads and at most 20GB of memory.

.. code-block:: bash

    snakemake -j 8 --resources mem_mb=20000

Cluster execution
-----------------
More commonly, the *CBPtools* workflow will be executed on a cluster. Upon creation of the project, a cluster.json file
is created in the `workdir`. This file must be edited so that all parameters are set properly. Commonly only the
`"__default__"` field needs to be edited to have the correct `account` name and `partition`, outlined below. The
default settings are inherited for each task and only overwritten if defined for the task.

.. code-block:: json

    {
        "__default__" :
        {
            "account" : "my account",
            "time" : "01:00:00",
            "n" : 1,
            "N" : 1,
            "c" : 1,
            "partition" : "core",
            "out" : "log/{rule}-%j.out",
            "name" : "unknown",
            "mem" : "1000M"
        }
    }

If you notice that jobs of certain tasks do not have enough time to finish or exceed the alotted memory (i.e., when the
memory estimation during the project setup is incorrect), they can be modified in the cluster.json file to be given
more time or memory. Note that the estimation tends to be liberal, so editing the cluster.json file to more accurately
define time and memory allocation may speed up the procedure. For more information on the cluster.json file, visit the
`Snakemake cluster configuration documentation <https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html#cluster-configuration>`_

The below example is for a cluster running the SLURM scheduler. The cluster configuration is used to fill in the
wildcards given for the -c argument (i.e., `{cluster.partition}` becomes 'core' in the above example of the
cluster.json file).

.. code-block:: bash

    snakemake -j 999 -w 240 -u cluster.json --resources -c "sbatch -p {cluster.partition} -n {cluster.n} -N {cluster.N} -t {cluster.time} -c {cluster.c} --mem-per-cpu={cluster.mem} --out={cluster.out} --job-name={cluster.name}"

The `-c "sbatch ..."` argument defines the command snakemake uses to start jobs through SLURM's sbatch command. The
`-w 240` is added to ensure that snakemake waits at least 240 seconds for files to appear on the file system. File
system latency may cause snakemake to assume something went wrong with generating the file, hence halting execution of
further tasks. If you find this to be the case, you may need to increase the waiting time. Note that the number of jobs
is set to 999 (`-j 999`), as Snakemake will let the SLURM scheduler manage job execution. This ensures that all jobs
for which the necessary input is available are placed in the queue.

File system latency may cause issues with particularly large input files. For instance, the `connectivity` task uses
time-series images as input, which can easily be as large as 5 GB. When all parallel job executions of this task try to
simultaneously load such big files, the jobs will slow one another down. For this purpose, the `io` resource has been
added to the rsfMRI and dMRI connectivity and probtrackx2 tasks. Simply add `io=x` to the `--resources` parameter,
where x is the number of simultaneous jobs you think your file system can handle. For example:

.. code-block:: bash

    snakemake -j 999 -w 240 -u cluster.json --resources io=10 -c "sbatch -p {cluster.partition} -n {cluster.n} -N {cluster.N} -t {cluster.time} -c {cluster.c} --mem-per-cpu={cluster.mem} --out={cluster.out} --job-name={cluster.name}"

Note that the timing of each task depends crucially on the current state of the cluster system and is therefore
difficult to predict. If you notice that jobs fail due to timeouts, the `time` parameter in `cluster.json` should be
increased.