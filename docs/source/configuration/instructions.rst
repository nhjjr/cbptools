.. _ConfigFile:

Configuration File
==================

Example Configuration
---------------------
Example configuration files are included in CBPtools. They contain default parameter settings and placeholder file
paths for the input data. Currently there are three types of input data supported: resting-state fMRI time-series data,
diffusion-weighted imaging data (as bedpostx output), and seed-by-target connectivity matrices. For precise
specifications and data requirements, read the respective instructions.

To get an example configuration file, run the following in the terminal. Allowed values for `--get` are: rsfmri, dmri,
or connectivity.

.. code-block:: bash

    cbptools example --get rsfmri

This copies an example resting-state configuration file to the current working directory.

Configuration File
------------------
The file is in the YAML format. Values can be added or modified, but must comply to the requirements. There are three
top-level keys in the file: `input_data_type`, `input_data` and `parameters`.

* `input_data_type`: The type of input data to expect. Allowed values are 'rsfmri', 'dmri', and 'connectivity',
* `input_data`: Within here all external input is defined. Read more `here <input_data>`_
* `parameters`: Contains all the parameters for the CBP processing. The steps are subdivided into `masking <params_masking>`_, `connectivity <params_connectivity>`_, `clustering <params_clustering>`_, and `summary <params_summary>`_.

Note that the configuration file is validated upon project creation. Errors occurring during validation cause the
project setup to fail. They can then be read in the log file placed inside of the selected working directory.
