.. _QuickStartGuide:

=================
Quick Start Guide
=================
If Python is available, *CBPtools* and its dependencies can quickly be :ref:`installed <installation>` using `pip`,
which is packaged together with Python.

.. code-block:: bash

    pip install cbptools

After installation, *CBPtools* can be called directly from the command line using the `cbptools` command. An example
configuration file can be obtained using the `example` directive:

.. code-block:: bash

    cbptools example --get modality

The `modality` argument should be replaced by the type of data you intend to give as input. Currently supported are
`rsfmri` (:term:`rsfMRI`), `dmri` (:term:`dMRI`), and `connectivity`. A configuration file with default and placeholder
settings will be created in the current working directory. Edit this file such that it correctly points to the input
data and has the desired :ref:`configuration values <GettingStartedSetup>`.

With the configuration file and input data in order, an :term:`rCBP` project can now be set up.

.. code-block:: bash

    cbptools create --config /path/to/config_file.yaml --workdir /path/to/workdir

Point the `--config` parameter to the configuration file and the `--workdir` to the directory in which the project
files will be placed. *CBPtools* will :ref:`validate <validation>` the configuration file and input data. If everything
is properly formatted, the project will be created in the designated working directory.

.. note::

    It is strongly suggested to read the log file, especially if there are any warnings or errors during the setup.

The next step is to go to the `workdir` and execute the workflow using `snakemake`, which is installed as a
dependency of *CBPtools*. Once this is done, the rCBP procedure will start.

.. code-block:: bash

    cd /path/to/workdir
    snakemake

Snakemake has a range of parameters it can be executed with. Common uses with *CBPtools* are outlined in the
:ref:`execution` section. For more customizability on snakemake :cite:`koster:2012`, visit the
`snakemake documentation <https://snakemake.readthedocs.io/en/stable/>`_.
