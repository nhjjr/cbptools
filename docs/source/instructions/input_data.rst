.. _InputData:

==========
Input Data
==========
The first step of configuring a regional CBP project is to define the `modality` of the input `data`. The files that
are required or allowed differ per modality. Currently supported modalities are resting-state functional MRI (rsfMRI)
and diffusion MRI (dMRI). It is also possible to provide connectivity matrices directly as input. This will skip the
steps in the workflow used to generate connectivity matrices from rsfMRI or dMRI data. All input data should be
quality controlled prior to using *CBPtools*, as only marginal validation is performed on the input data. Faulty data
may halt processing, but in a worst-case scenario such data may provide untrustworthy output.

The input data is separated into modality-independent and modality-dependent categories.

********************
Modality-independent
********************
Modality-independent input data includes (1) a binary 3-dimensional ROI file in the NIfTI image data format, and (2)
an optional 3-dimensional target mask in the same data format, used to define the connections that are considered for
each ROI voxel. If not provided by the user, a modified FSL (http://www.fmrib.ox.ac.uk/fsl/) distributed average
Montreal Neurological Institute (MNI) 152 T1 whole-brain gray matter group template (2mm isotropic) will be used as the
target. In this case, the input data should match the same MNI152 template as well. If connectivity matrices are
entered directly, the target mask will not be used. Lastly, (3) a participants file as a tab-separated text file with a
column called 'participant_id' containing all unique identifiers of hte subjects to be included in this project.

.. glossary::
   seed_mask (*str*)
      A binary region-of-interest NIfTI image in the same space as the time-series and target mask. The extension must
      be either `.nii` or `.nii.gz`.

   target_mask (*str*)
      A binary NIfTI image covering a target region (e.g., the whole brain). If left empty, the MNI152 2mm gray-matter
      mask will be used by default. The extension must be either `.nii` or `.nii.gz`.

   participants
      A tabular file containing a column with participant IDs.

         file (*str*)
            Path to the participants file.
         delimiter (*str*)
            Delimiter to use (e.g., '\t' for tab-, or ',' for comma-delimited fiels). By default this is set to '\t'.
         index_column (*str*)
            Name for the column containing the participant IDs. By default this is set to 'participant_id'.

      When a `{participant_id}` template is requested, then the file path must contain this template at the place where
      otherwise the participant ID would go. This template will be replaced by the actual participant IDs during
      execution of the pipeline.

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

******************
Modality-dependent
******************
Modality-dependent input data differs between the rsfMRI and dMRI modalities, as well as when the connectivity matrices
are entered directly.

Resting-State Data
==================
A 4-dimensional time-series NIfTI image per subject (as defined in the participants file) is required, optionally
accompanied by a tab-separated text file containing confounds for each time point as columns. *CBPtools* assumes that
the rsfMRI data has been treated with necessary fMRI preprocessing including realignment and normalisation to a
template space. If the default target mask is used, then the template space must be MNI152 with 2mm isotropic voxels.
Denoising based on independent component analysis like Automatic Removal of Motion Artifacts (ICA-AROMA)
:cite:`pruim:2015` or FMRIB's ICA-based X-noiseifier (FIX) :cite:`salimi:2014` is encouraged if suitable.

These fields will be used when the `modality` is set to 'rsfmri':

.. glossary::

   time_series (*str*)
      Path to a 4D NIfTI image containing the resting-state time-series (x, y, z, timepoints). The time-series shape
      and affine must match that of the seed and target masks. This field must contain the `{participant_id}` template,
      as there should be one time-series image per subject. The extension must be either `.nii` or `.nii.gz`.

   confounds (*str, optional*)
      A delimited (e.g., .tsv for tab-delimited) file with a confound signal per columns and a 1-line header. Columns
      can be selected using the `columns` parameter. The column length must match the length of the timepoints in the
      signal time-series. This field has sub-fields:

      file (*str*)
         Path to the confounds file
      delimiter (*str*)
         Delimiter to use (e.g., '\t' for tab-, or ',' for comma-delimited fiels). By default this is set to '\t'.
      columns (*list*)
         List of columns that should be used. If left empty, all columns are used. Otherwise only the selected columns
         will be used in nuisance signal regression. A glob pattern (\*) can be used to select multiple columns that
         match the expression (e.g., 'motion-\*' includse 'motion-x', 'motion-y', motion-z', etc.).


Diffusion-Weighted Imaging Data
===============================
*CBPtools* uses probabilistic diffusion tractography to generated connectivity matrices for the dMRI modality.
Therefore, input necessary to perform FSL's probabilistic diffusion tractography (PROBTRACKX2) is required per subject,
consisting of: (1) Outputs from Bayesian Estimation of Diffusion Parameters Obtained using Sampling Techniques
(BEDPOSTX), (2) a brain extraction (BET) binary mask file, (3) a transform file taking seed space to DTI space
(either a FLIR matrix or FNIR warpfield), and (4) a file describing the transformation from DTI space to seed space.
Each of these files is subject-specific and can be obtained from FSL's BEDPOSTX output.

These fields will be used when the `modality` is set to 'dmri':

.. glossary::

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
=================
When connectivity is used as input rather than resting-state or difussion-weighted imaging data,

Connectivity matrices may be provided as source input in lieu of rsfMRI or dMRI data. The `masking` and `connectivity`
tasks are then skipped. This means that connectivity matrices must be presented in the way *CBPtools* generates them.
They must be provided in an ROI-voxel by target-voxel shape as a NumPy array. This can either be done in the
uncompressed .npy format, or as a compressed .npz file. In case of the latter, make sure that the array is stored as
`connectivity.npy` inside of the .npz archive. Along with the connectivity matrix, a binary 3-dimensional mask of the
ROI in NIfTI image data format is expected. The number of voxels in this mask must coincide with the number of seed
voxels on the first dimension of the connectivity matrix. Lastly, a NumPy array (.npy) of seed voxel coordinates must
be provided in the order the voxels are represented in the connectivity matrix. This is used to map the clustering
results onto the ROI mask for visualization purposes.

These fields will be used when the `modality` is set to 'connectivity':

.. glossary::

   connectivity_matrix (*str*)
      The path to a seed by target connectivity matrix. The number of seed voxels on the first dimension must match the
      number of seed voxels in the seed mask image. The order in which the seed voxels are lsited along the first axis
      depends on the order that was used to extract the voxels from the mask (i.e., F- or C-contiguous order).
      *CBPtools* connectivity matrices have the seed voxels in C-order. This field must contain the `{participant_id}`
      template. The extension must be either `.npy` or `.npz`. If the compressed .npz format is used, the array in the
      archive must be named `connectivity.npy`.

   seed_coordinates (*str*)
      The path to a 2-dimensional NumPy array file with shape seed voxels by 3. The file contains the 3D coordinates of
      each seed voxel in the order that the seed voxels appear in the connectivity matrix. The extension must be `.npy`.