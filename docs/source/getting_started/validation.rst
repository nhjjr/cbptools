.. _validation:

==========
Validation
==========
Validation consists of two steps: (1) validation of the configuration file, and (2) validation of the data set. If the
configuration file is not properly formatted in the YAML structure or contains values that do not meet the necessary
requirements, the setup procedure will fail and create a log file containing the reasons why the setup failed. The
data set is only marginally validated, so it is not recommended to rely on it to ensure that the input data has no
flaws. This validation step takes a bit longer, as each external file is checked if it meets the necessary criteria
to be processed by CBPtools. At this point subjects may be excluded from the project if their data does not pass
validation. CBPtools will provide warnings when this happens in the log file.

.. note::
    It is strongly recommended to read the log file after the setup completes (or fails)

Configuration File
==================
The configuration file is validated against a schema (on GitHub, it is the `cbptools/schema.yaml` file), which
contains all available data and parameter fields and their validation rules. The following rules are available:

:required: If a field is required, checks whether the field has a value. If no value is given and a default value is
    present, the default value will be used.

:dependency: Some fields have conditional requirements, i.e., the field is only used if a certain condition is met,
    such as another field having a specific value. For example, a field which depends on the modality to be 'rsfmri'
    will only be used if that modality is set. Note that `required` fields are only required if the dependency is met.

:type: Checks if the given value's type matches the expected type. For example, when 'integer' is the expected type,
    then a number has to be entered. If 'list[string]' is set as an expected type, a list of strings (e.g., plain text)
    must be given in a YAML recognisable format (i.e., a valid list[string] would be `[foo, bar, baz]`).

:contains: Checks if a value (i.e., list or string) contains a required part. For example, data set files that are
    subject specific must contain the `{participant_id}` wildcard.

:allowed: Checks if the field consists only out of allowed values, which are represented as a list. For
    example, when selecting a clustering algorithm, the field's allowed values are [kmeans, spectral, hierarchical].

:min: Checks if the field is above the minimum allowed value. This only works with integer and float type values.

:max: Checks if the field is above the maximum allowed value. This only works with integer and float type values.

:minlength: Checks if the field is above a minimum required length. For strings, this means the number of characters
    must be above this value. For lists and other iterables, this means the number of items must be above this value.

:maxlength: As minlength, but instead checks if the field is below a maximum length.

Some fields have custom rules that only apply to that particular field. As of the current version, these rules are:

:(custom) bandpass: This applies to the `parameters:connectivity:bandpass.band` field and checks whether the high-pass
    value is smaller than the low-pass value.

:(custom) voxel dimensions: This applies to the voxel_dimensions field for up- and downsampling and checks whether
    the field has exactly 1 or 3 values. In the case of 1 value, it will be used for all 3 dimensions.

:(custom) repetition-time: This applies to the `parameters:connectivity:bandpass.tr` field and warns if the value is
    larger than 100. Since repetition time is specified in seconds, a value larger than 100 is likely to be a mistake.
    This triggers only a warning, however, and may be ignored if the value is not a mistake.

:(custom) agglomerative linkage: When using agglomerative clustering, if linkage is set to 'ward' then this rule checks
    whether distance is set to 'euclidean', as ward linkage requires euclidean distance.

:(custom) has sessions: Checks whether the input data fields contain the '{session}' wildcard if sessions are defined.
    If no sessions are used, then the wildcard should not be present.

:(custom) space match: When `data:masks:space` is set to 'native', then this rule checks if all masks used are
    subject specific (i.e., contain the '{participant_id}' wildcard). It furthermore checks if the target mask field
    is defined, because that mask also has to be subject specific.

:(custom) benchmarking: If benchmarking is to be performed, the `psutil` package must be installed. This rule checks if
    the package is installed.

:(custom) spectral kernel: When using the 'precomputed' option for defining the kernel with spectral clustering, the
    modality must be 'connectivity'. CBPtools then assumes that the connectivity input data consists of similarity
    matrices rather than connectivity matrices.

:(custom) reference: When reference images are defined, median filtering cannot be used to preprocess the seed mask. If
    median filtering is used, this might change the shape of the seed mask so that it no longer matches the reference
    images and thus comparison between the cluster-labeled seed mask and reference images becomes impossible.

.. _validationInputData:

Input data
==========
The input data is minimally validated after the configuration file has passed validation. Note that when referencing
files in the configuration, relative paths are automatically converted to absolute paths. It is always best to specify
absolute paths, so that the configuration file can be used regardless of its location. Each input data file is
validated as follows:

:participants: The participants file is read using the `pandas` package. The given delimiter and index column is used
    to obtain a list of the participant IDs. If the file cannot be found or read, or if there are no participant IDs,
    the validation will fail as the participant IDs are necessary to validate subject-specific files.

:seed mask: The seed mask is loaded using the `nibabel` package. If a `region_id` is specified, it is treated as an
    atlas and a (composite) mask will be extracted using the region IDs. If the mask contains any NaN or inf values,
    the setup will fail.

:target mask: If no target mask is given, a default 2mm MNI152 gray-matter mask will be used. Note that the seed mask
    and time-series must still match this mask, hence by not specifying a target mask it is automatically assumed that
    all data is in 2mm MNI152 space. If a target mask is defined, it will get the same validation as the seed mask
    except it cannot be treated as an atlas.

:(rsfmri) time-series: For each subject in the participants file, the time-series image header is inspected. The image
    must have the same space as the seed and target masks and must have 4-dimensions (x, y, z, timepoints). The
    `nibabel` package is used for loading the image. If the file cannot be found or read as a NIfTI image, the subject
    will be marked as having unusable data.

:(rsfmri) confounds: The confounds file is read using the `pandas` package. The given delimiter is used to read the
    file, and if columns are specified they must be present in the file. If no columns are given, all columns will be
    used. Furthermore, the number of rows (excluding the header row) must match the number of timepoints in the
    time-series of that subject. If any of these checks fail, the subject will be marked as having unusable data and
    will be excluded from the project.

:(dmri) samples: Unlike other input data, the samples are a collection of files. Therefore, using the `glob` package,
    an asterisk (*) is appended to the end of the given path and the number of files matching this pattern should be
    at least 1.

:(dmri) bedpostX files: All other dMRI input data is loaded using the `nibabel` package. If this fails, the subject is
    marked as having unusable data.

:(connectivity) connectivity matrices: The connectivity matrices must be in the NumPy .npy or .npz format. In case the
    .npz format is used, the matrices must be saved under the key 'connectivity'. Furthermore, the length of the
    x-axis must match the number of voxels in the seed mask. Failing to meet these criteria will result in the subject
    being marked as having unusable data. Note that at this step, only the NumPy header information is being used, which
    significantly speeds up the validation procedure.

:(connectivity) seed coordinates: The seed coordinates are a NumPy .npy array where the x-axis must match the length
    of the number of voxels in the seed mask, and the y-axis must be of length 3 (x, y, and z coordinates).

:reference images: These images are loaded using the `nibabel` package and compared to the seed mask, where the exact
    same voxels need to be used. Furthermore, the reference images must contain at least two clusters.


Subjects marked as having unusable data are excluded from the project. If during the data validation there are fewer
than 2 subjects remaining, the setup will fail. If native space masks are being used, then the setup will fail if there
is not at least 1 subject remaining.

If multi-session data is used, then these checks will apply to all the given sessions. A subject will be excluded from
the project even if only one session contains unusable data.

Creating the project
====================
Once the input data validation procedure has completed, a memory and disk space estimate is made using the input data.
This is a very liberal estimate. When executing the workflow on a cluster system that requires each job to specify the
memory that it will need, these values are used. If the modality is set to `dmri`, the availability of FSL and
`probtrackx2` is checked. If the tool is not available, a warning will be given.

Next, the workflow is built using only the necessary tasks (i.e., the task for spectral clustering is not used when
another clustering algorithm is chosen, and additional tasks are necessary to deal with multi-session data). This
workflow is stored as `Snakefile` in the project directory.

A `cluster.json` file is added to the project directory, which can be used when submitting jobs to a scheduler. This
file is used by snakemake to define cluster parameters for each rule. When using a scheduler (e.g., SLURM or qsub) this
file defines timing, account name, cluster name, etc. For more information, read the
`snakemake guidelines <https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html#cluster-configuration>`_

The participants that are included in the study are stored in `participants.tsv`, whereas participants that were excluded
are stored in `participants_bad.tsv`. The `participants.tsv` can be edited after the setup by removing or adding
participants. The index column in this file should always be named 'participant_id'.

The seed and target masks are stored as `seed_mask.nii.gz` and `target_mask.nii.gz` respectively. For the 'dmri'
modality, an additional `highres_seed_mask.nii.gz` is included, which is the seed mask stretched (not upsampled!) to a
higher resolution. A `seed_coordinates.npy` file is created (or copied, if the modality is 'connectivity') containing
the x, y, and z coordinates of each seed voxel in C-order.
