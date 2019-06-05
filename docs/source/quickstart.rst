Quick Start Guide
=================

Quick example
-------------
After installing CBPtools, an example configuration file can be obtained.

.. code-block:: bash

    cbptools example --get data_type

Where data_type is replaced by the data you intend to give as input, either rsfmri, dmri, or connectivity. A
configuration file with default and placeholder settings will now appear in your current working directory. Edit this
file so that it correctly points to the input data and all parameters are set accordingly.

A CBP project can now be created by specifying the location to the configuration file and the working directory in
which the project will be created.

.. code-block:: bash

    cbptools create --config /path/to/config_file.yaml --workdir /path/to/workdir

If any errors appear project creation will fail, and an error log will appear inside of the working directory. If no
errors occur, the project will be created. Change directory to the project directory and execute the `Snakefile` using
snakemake, which is installed as a dependency of CBPtools.

.. code-block:: bash

    cd /path/to/workdir
    snakemake

For more customizability on snakemake (Köster and Rahmann, 2012), visit the
`snakemake documentation <https://snakemake.readthedocs.io/en/stable/>`_.

References
----------
Köster J, Rahmann S (2012): Snakemake--a scalable bioinformatics workflow engine. Bioinformatics 28:2520–2522.