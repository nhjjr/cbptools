.. _instructions:

=============
Configuration
=============

*********************
Example Configuration
*********************
An example configuration file with placeholders and default parameters can be created using the `cbptools example`
command in the terminal.

.. code-block:: bash

    cbptools example --get modality

The `modality` argument is replaced by the type of data you intend to give as input, either `rsfmri`, `dmri`, or
`connectivity`. The configuration file will appear in the current working directory and *has* to be edited such that
it correctly points to the input data and has the desired parameter values.

******************************
Editing the Configuration File
******************************
Configuration files are in the YAML format and must therefore use the YAML syntax. The example configuration file
contains all available fields for the requested modality, either having default or placeholder values. There are three
top-level keys in the file: `modality`, `data`, and `parameters`.

* `modality`: *CBPtools* will expect this type of input data and handle validation and setup accordingly. Allowed
  values are rsfmri, dmri, and connectivity.
* `data`: Within this field all external input to the workflow is defined. Read more `here <input_data>`_
* `parameters`: Contains all the parameters for regional CBP processing as outlined in the `workflow <workflow>`_. The
  steps are subdivided into `masking <params_masking>`_, `connectivity <params_connectivity>`_,
  `clustering <params_clustering>`_, and `report <params_report>`_.

The configuration file is validated during the `setup <setup>`_. Errors occurring during validation cause the setup to
fail. The resulting log file can then be used to find and resolve errors.