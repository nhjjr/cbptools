==================
Validation & Setup
==================
With a properly defined configuration file and the input data in place and quality controlled, the set up procedure can
be started to create a regional CBP project folder.

.. code-block:: bash

    cbptools create --config /path/to/config_file.yaml --workdir /path/to/workdir

The setup will first validate the YAML configuration file, ensuring that all parameters and input data is defined
properly. Next, the input data will be validated. This validation is only marginal to avoid common mistakes. It is not
something to rely upon for quality controlling the input data, this is beyond the scope of *CBPtools*. Once the
validation has succeeded, the seed- and (optionally) target masks will be preprocessed according to the given
parameters.

If at any time during the setup an error occurs, it will be logged and the setup procedure will halt. Only if the setup
passes without errors (i.e., critical mistakes in the configuration file or input data) will the project be created in
the given working directory (`--workdir` parameter). Warnings should not be ignored. They are also given when default
settings not otherwise specified in the configuration file are used.

Upon completion of the setup, the project directory will contain the following files:

:cluster.json: Used by snakemake to define cluster parameters for each rule. When using a cluster environment (e.g.,
   SLURM or qsub) this file defines timing, account name, cluster name, etc. For more information, read the
   `snakemake guidelines <https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html#cluster-configuration>`_

:participants.tsv: The input participants file is converted to a .tsv format (tab-delimited). The index column is always
   named 'participant_id' and contains all participants for which the data is included in this project. If at any time
   a participant is to be removed, this file can be edited. Be careful when adding participants this way, because
   no validation is performed after creating the project.

:participants_bad.tsv: This file contains all the participant indices that were removed during project creation. These
   are participants for which either data was missing, or the data did not meet the requirements. Note that this file
   is purely informative. It is not used anywhere in the regional CBP procedure.

:log/project_<timestamp>.log: This log file contains information about the setup process.

:seed_mask.nii: Binary region-of-interest mask after processing

:target_mask.nii: Binary target mask after processing (default is the 2mm MNI152 gray matter mask). This mask will not
   appear for the 'connectivity' `modality` because it is not necessary.

:highres_seed_mask.nii: seed mask stretched to a higher resolution. This mask is only used for the 'dmri'
   `modality`

:seed_coordinates.npy: These are the indices (3D 'coordinates') of all seed voxels in a NumPy 2D array. The order of the
   indices is determined by the order in which the seed mask will be extracted (F- or C-contiguous). This file must be
   provided as input in case connectivity matrices are used as input.

:Snakefile: The workflow for the processing pipeline, used by snakemake to initiate it. This file also contains all
   external and internal input files as well as parameters used during project creation.
