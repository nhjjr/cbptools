.. _ExampleData:

============
Example Data
============
A subset of the preprocessed data used in the *CBPtools* paper has been made available online. The  :term:`rsfMRI` and
:term:`dMRI` data for 100 randomly drawn subjects out of the 300 subjects in total can be downloaded. Furthermore
included are the three :term:`ROI` NIfTI images, as well as the *CBPtools* configuration files that were used to
process the data.

Get the data
============
The example data set was prepared and uploaded using DataLad version 0.12.0rc6 and has a total size of **243 GB**. It
can be downloaded using DataLad, which can be installed using *apt-get* or *pip*. Visit the
:ref:`DataLad website <https://www.datalad.org/get_datalad.html>` for more information on how to install the tool. For
more information on how to use DataLad, visit the
:ref:`online documentation <http://docs.datalad.org/en/latest/index.html>`.

The example data is located on a remote location linked to through a
:ref:`GitHub repository <https://github.com/inm7/cbptools-example-data>`. With DataLad installed, it can be downloaded
as follows:

.. code-block:: bash

    datalad install --get-data --source https://github.com/inm7/cbptools-example-data.git

This will obtain all the meta data from the GitHub repository and then download the data from the remote.

Alternatively it is possible to download only parts of the data. The command below will only obtain the meta data,
which is considerably smaller than the actual data.

.. code-block:: bash

    datalad install https://github.com/inm7/cbptools-example-data.git

It is now possible to look at the structure of the data set and other included files (which are all symlinks that are
currently broken, as the real data has not been downloaded yet). For example, to download only the gray matter mask, it
is possible to run the following command *inside* the example data directory (by default this is
`cbptools-example-data`).

.. code-block:: bash

    datalad get gray_matter.nii.gz

It is possible to use patterns in order to download the data. For example, if you only want to download the rsfMRI
data, then the following command will do just that:

.. code-block:: bash

    datalad get data_set/*/rsfmri

Using the example configuration
===============================
A *CBPtools* project can be created using any of the provided configuration files or a custom configuration file. This
example will outline the use of the preSMA-SMA ROI with the rsfMRI data (i.e., the `config_r_presma-sma_rsfmri.yaml`
configuration file), although it can be substituted by any other configuration file to use different settings and data.

The code block below will download the necessary data using DataLad and then create a *CBPtools* project.

.. code-block:: bash

    datalad install https://github.com/inm7/cbptools-example-data.git
    cd cbptools-example-data
    datalad get data_set/*/rsfmri
    datalad get data_set/participants.tsv
    datalad get config_r_presma-sma_rsfmri.yaml
    datalad get gray_matter.nii.gz
    datalad get r_presma-sma.nii.gz
    cbptools create --config config_r_presma-sma_rsfmri.yaml --workdir cbptools/r_presma-sma_rsfmri

The `--workdir` parameter is used to define where the project files (and eventual output data) will be stored. This can
be any directory on the file system with read and write access. The snippet above will place it inside of the data
directory in the `cbptools/r_presma-sma_rsfmri` folder.

Any errors and warnings occurring during the setup will be logged, which is available either in the current directory
(if the setup fails) or in the log folder inside the `workdir` (if the setup succeeds). If there are any errors, the
project will not be created until they are resolved, otherwise the setup is complete and the processing can be started.

Change directory to the `workdir` and execute the workflow (contained in the `Snakefile`) using Snakemake, which is
installed as a dependency of *CBPtools*.

.. code-block:: bash

    cd cbptools/r_presma-sma_rsfmri
    snakemake

Snakemake has a range of parameters it can be executed with. Common uses with *CBPtools* are outlined in the
:ref:`execution` section. For more customizability on snakemake :cite:`koster:2012`, visit the
`snakemake documentation <https://snakemake.readthedocs.io/en/stable/>`_.

Looking at the results
======================
The results will be placed inside of the `workdir`, which in the above example is `cbptools/r_presma-sma_rsfmri`. If
benchmarks are available, they will be in the `benchmarks` folder. All log files are in the `log` folder. The group
results are in the `group` folder, which contains folders for each requested number of clusters, as well as figures and
tables. The `individual` folder will contain interim results for each subject. If plots and metrics for individual
subjects are requested in the configuration file, they will also appear here.
