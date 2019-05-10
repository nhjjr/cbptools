Input Validation
================
Upon the creation of a project, following the set up of the configuration file, all input data and parameters will be
validated to see whether they meet the requirements (outlined in the :ref:`ConfigFile`). Only if validation passes
without errors (critical mistakes in the configuration file or input data) will the project be created. If there are
errors they are outlined in the log file which accessible from the chosen `workdir`. All errors must be addressed
before the project will be set up properly. Furthermore, warnings should not be ignored!

Once validation passes mask files are processed using the parameters for the `masking` task. The resulting mask files
are saved to the `workdir`. Once completed, the working directory will have the following files:

:cluster.json: Used by snakemake to define cluster parameters for each rule. When using a cluster environment (e.g.,
   SLURM or qsub) thisf ile defines timing, account name, cluster name, etc. For more information, read the
   `snakemake guidelines <https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html#cluster-configuration>`_

:participants.tsv: The input participants file is converted to a .tsv format (tab-delimited). The index column is always
   named 'participant_id' and contains all participants for which the data is included in this project. If at any time
   a participant is to be removed, this file can be edited. Be careful when adding participants this way, because
   no validation is performed after creating the project.

:participants_bad.tsv: This file contains all the participant indices that were removed during project creation. These
   are participants for which either data was missing, or the data did not meet the requirements.

:log/project_<timestamp>.log: This log file contains information about the project creation.

:seed_mask.nii: Binary region-of-interest mask after processing

:target_mask.nii: Binary target mask after processing (default is the 2mm MNI152 gray matter mask). This mask will not
   appear for the 'connectivity' `input_data_type` because it is not necessary.

:highres_seed_mask.nii: seed mask stretched to a higher resolution. This mask is only used for the 'dmri'
   `input_data_type`

:seed_indices.npy: These are the indices (3D 'coordinates') of all seed voxels in a NumPy 2D array. The order of the
   indices is determined by the order in which the seed mask will be extracted (F- or C-contiguous). This file must be
   provided as input in case connectivity matrices are used as input.

:Snakefile: The workflow for the processing pipeline, used by snakemake to initiate it. This file also contains all
   external and internal input files as well as parameters used during project creation.
