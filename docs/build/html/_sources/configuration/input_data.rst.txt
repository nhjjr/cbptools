.. _InputData:

Input Data
==========
The first step of configuring a CBP project is to define your input data. Depending on the `input_data_type` this has
different entries. Some `input_data` fields have sub-fields. In that case, the key itself may contain other key:value
pairs. The file path normally entered at the top field may then also be entered in the sub-field called *file*. This
sub-field is required only when sub-fields are used at all. Sub-fields are used to define options for the input data,
such as the separator to be used for reading tabular data (e.g., "\t" for tab-separated (.tsv) files).

Valid examples:

.. code-block:: yaml

    input_data:
        time_series: 'path/to/{participant_id}/time_series.nii'
        seed_mask: 'path/to/seed_mask.nii

.. code-block:: yaml

    input_data:
        time_series:
            file: 'path/to/{participant_id}/time_series.nii'
            validate_shape: true
        seed_mask: 'path/to/seed_mask.nii'

Participants Data
-----------------
.. glossary::
   participants (*str*)
      A tabular file containing a column with participant IDs. By default this column should be named
      'participant_id'. This field has sub-fields:

         file (*str*)
            Path to the participants file.
         sep (*str*)
            Delimiter to use.
         index_col (*str*)
            Alternative name for the column containing the participant IDs

      When a `{participant_id}` template is requested, then the file path must contain this template at the place where
      otherwise the participant ID would go. This template be replaced by the actual participant IDs during execution
      of the pipeline.

      For example, the following participants file is used:

      ============== ====== ====
      participant_id gender age
      ============== ====== ====
      571144         M      25
      517239         M      23
      812746         M      22
      213421         F      26
      204622         F      30
      ============== ====== ====

      Then, the file path `/path/to/{participant_id}/time_series.nii` will be converted to
      `/path/to/571144/time_series.nii`, `/path/to/517239/time_series.nii`, and so on.


Resting-state fMRI Data
-----------------------
This section is useful only if the `input_data_type` is set to 'rsfmri'.

.. glossary::
   seed_mask (*str*)
      A binary region-of-interest NIfTI image in the same space as the time-series and target mask. The extension must
      be either `.nii` or `.nii.gz`.

   target_mask (*str*)
      A binary NIfTI image covering a target region (e.g., the whole brain). If left empty, the MNI152 2mm gray-matter
      mask will be used by default. The extension must be either `.nii` or `.nii.gz`.

   time_series (*str*)
      4D NIfTI image containing the resting-state time-series. This field must contain the `{participant_id}` template.
      This field has sub-fields:

      file (*str*)
         Path to the time-series file
      validate_shape (*bool, optional*)
         If `true` the validation procedure will evaluate whether the time-series files match in shape and affine to
         the seed mask. This can be time consuming depending on the number of participants and the size of the
         time-series images.

   confounds (*str, optional*)
      A tabular confounds file containing columns of nuisance signal regressors. This field has sub-fields:

      file (*str*)
         Path to the confounds file
      sep (*str*)
         Delimiter to use
      usecols (*list*)
         List of columns that should be used. If left empty, all columns are used. Otherwise only the selected columns
         will be used in nuisance signal regression. A glob pattern (\*) can be used to select multiple columns that
         match the expression (e.g., 'motion-\*' includse 'motion-x', 'motion-y', motion-z', etc.).


Diffusion MRI Data
------------------
This section is useful only if the `input_data_type` is set to 'dmri'.

.. glossary::
   seed_mask (*str*)
      A binary region-of-interest NIfTI image in the same space as the time-series and target mask. The extension must
      be either `.nii` or `.nii.gz`.

   target_mask (*str*)
      A binary NIfTI image covering a target region (e.g., the whole brain). If left empty, the MNI152 2mm gray-matter
      mask will be used by default. The extension must be either `.nii` or `.nii.gz`.

   bet_binary_mask (*str*)
      File path to a `BET <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#bet>`_ binary mask file. This field must
      contain the `{participant_id}` template. The extension must be either `.nii` or `.nii.gz`.

   xfm (*str*)
      Transform taking seed space to DTI space (either FLIRT matrix or FNIRT warpfield). This field must contain the
      `{participant_id}` template. The extension must be either `.nii` or `.nii.gz`.

   inv_xfm (*str*)
      Transform taking DTI space to seed space. This field must contain the `{participant_id}` template. The extension
      must be either `.nii` or `.nii.gz`.

   samples (*str*)
      Merged samples derived from bedpostX output. This field must contain the `{participant_id}` template. Note that
      this is the same file path that would otherwise be entered in FSL. This it selects all files that start with the
      entered file path (e.g., /path/to/samples/merged will take all files starting with merged).


Connectivity Data
-----------------
This section is useful only if the `input_data_type` is set to 'connectivity'.

.. glossary::
   seed_mask (*str*)
      A binary region-of-interest NIfTI image in the same space as the time-series and target mask. The extension must
      be either `.nii` or `.nii.gz`.

   seed_indices (*str*)
      A numpy array file containing the 3D coordinates of each seed voxel in the order that the seed voxels are
      represented in the connectivity matrix. The extension must be `.npy`.

   connectivity_matrix (*str*)
      A numpy array file containing a connectivity matrix of shape (seed, target), where the seed voxels must match the
      amount of voxels in the seed mask. This field must contain the `{participant_id}` template. The extension must be
      `.npy`.
