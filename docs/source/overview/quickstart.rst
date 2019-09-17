#################
Quick Start Guide
#################
When *CBPtools* is installed it can be called directly from the command line using the `cbptools` command. An example
configuration file can be obtained using the following line:

.. code-block:: bash

    cbptools example --get modality

The `modality` argument is replaced by the type of data you intend to give as input, either `rsfmri`, `dmri`, or
`connectivity`. A configuration file with the default and placeholder settings will now appear in your current working
directory. The file *has* to be edited such that it correctly points to the input data and has the desired parameter
values.

With the configuration file and input data in order, a regional CBP project can now be set up.

.. code-block:: bash

    cbptools create --config /path/to/config_file.yaml --workdir /path/to/workdir

The `--config` parameter is used to define the path to the configuration file. The `--workdir` parameter is the
directory in which the project files will be placed.

Both the configuration file and input data will now be validated. Note that data validation is rudimentary in order to
prevent common mistakes, but it is strongly recommended to manually ensure the given input data is correct. If any
errors occur during this process, an error log will appear in the current working directory. Take note of the errors
and resolve them to continue (i.e., rerun the `cbptools create` command after the errors are addressed). If no errors
occur, the project will be created in the designated working directory. Change directory to the working directory and
execute the `Snakefile` using snakemake, which is installed as a dependency of *CBPtools*.

.. code-block:: bash

    cd /path/to/workdir
    snakemake

For more customizability on snakemake :cite:`koster:2012`, visit the
`snakemake documentation <https://snakemake.readthedocs.io/en/stable/>`_.
