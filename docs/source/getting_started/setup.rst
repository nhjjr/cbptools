.. _GettingStartedSetup:

=====
Setup
=====
The first step to creating a CBPtools project is to write a configuration file. This file contains the links to the
:ref:`data set <ConfigurationInputData>` you intend to use, as well as the :ref:`parameters <ConfigurationParameters>`
for the tasks that will process the data.

Example Configuration
=====================
An example configuration file can be obtained using the `example` directive. This file will have a set of parameters
in the proper format to use as a configuration file. Note that before use, the the values have to be changed to point
to the input data.

.. code-block:: bash

    cbptools example --get modality

The `modality` argument should be replaced by the type of data you intend to give as input. Currently supported are
`rsfmri` (:term:`rsfMRI`), `dmri` (:term:`dMRI`), or `connectivity`. A configuration file with default and placeholder
settings will be created in the current working directory.

Note that not all parameter fields are represented in this example configuration file. All available parameters fields
are listed :ref:`here <ConfigurationParameters>`, and all available input data fields are listed
:ref:`here <ConfigurationInputData>`.

Editing the Configuration File
==============================
Configuration files are in the YAML format and must therefore use the YAML syntax. The example configuration file
contains all available fields for the requested modality, either having default or placeholder values. There are three
top-level keys in the file: `modality`, `data`, and `parameters`.

* `modality`: *CBPtools* will expect this type of input data and handle validation and setup accordingly. Allowed
  values are `rsfmri`, `dmri`, and `connectivity`.
* `data`: Within this field all external inputa to the workflow are defined. Read more :ref:`here <ConfigurationInputData>`
* `parameters`: Contains all the parameters for :term:`rCBP` processing as outlined in the :ref:`workflow <workflow>`.
  Read more :ref:`here <ConfigurationParameters>`

Running the setup
=================
With a properly defined configuration file and the input data in place and quality controlled, the setup procedure can
be started to create a CBPtools project folder.

.. code-block:: bash

    cbptools create --config /path/to/config_file.yaml --workdir /path/to/workdir

The `--config` (alternative `-c`) parameter is used to define the path to the configuration file. The `--workdir`
(alternative `-w`) parameter is the directory in which the project files will be placed. This command will immediately
trigger the :ref:`validation <validation>` of the configuration file and input data. If this succeeds, a project folder
will be created at the `--workdir` location (i.e., /path/to/workdir in this case).
