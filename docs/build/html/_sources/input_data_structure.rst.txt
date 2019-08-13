Input Data Structure
====================
Necessary source input depends in part on the modality of the dataset. Currently supported modalities are rsfMRI and
dMRI. Modality-independent input data includes (1) a binary ROI file in the 3-dimensional NIfTI image data format, and
(2) an optional 3-dimensional target mask in the same data format, used to define the connections that are considered
for each ROI voxel. If not provided by the user, the `FSL <http://www.fmrib.ox.ac.uk/fsl/>`_ distributed Montreal
Neurological Institute (MNI) whole-brain gray matter group template will be used as the target. Note that this
necessitates the required input data to be in the MNI152 template space as well.

For rsfMRI data, a 4D time-series NIfTI image must be provided per subject, optionally accompanied by a tab-separated
text file containing confounds for each time point as columns. CBPtools requires rsfMRI data to be treated with
standard fMRI preprocessing including realignment and normalisation to a template space (e.g. MNI152) in NIfTI format.
If the default target mask is used, then the voxel size should be 2 mm isotropic. Denoising based on independent
component analysis like Automatic Removal of Motion Artifacts (ICA-AROMA) (Pruim et al., 2015) or FMRIB's ICA-based
X-noiseifier (FIX) (Salimi-Khorshidi et al., 2014) is encouraged. In particular, FIX has been shown to work well in
combination with mean white matter and cerebrospinal fluid signal regression (Plachti et al., 2019). The dMRI modality
requires input necessary to perform FSL’s probabilistic diffusion tractography (probtrackx2), consisting of: (1)
bedpostx sampling results,  (2) a brain extraction (BET) binary mask file, (3) a transform file taking seed space to
DTI space (either a FLIR matrix or FNIR warpfield), and (4) a file describing the transformation from DTI space to seed
space. Each of these files is subject-specific and can be obtained from FSL’s bedpostx command.

All input data should be quality controlled prior to using CBPtools, as only marginal validation is performed on the
input data. Faulty data may halt processing until the issues are resolved, but in a worst-case scenario such data may
provide untrustworthy output. Further specified during the setup are options to modify the connectivity measures, the
clustering parameters (e.g., the range of k clusters requested) and validity measures, as well as the desired output
file formats.

Resting-State Data
------------------

:Signal time series: The time-series must be in a 4D NIfTI format (x, y, z, timepoints) for each subject. The
   time-series shape and affine must also match that of the seed- and target masks.

:Confounds time series: *(optional)* A delimited (e.g., .tsv for tab-delimited) file with signals as columns and a
   1-line header. Columns can be selected using the `usecols` parameter (see :ref:`InputData`). The column length must
   match the length of the timepoints in the signal time series.


Diffusion-Weighted Imaging
--------------------------

:BET binary mask: Each subject must have a `BET <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#bet>`_ binary
   mask file.

:XFM: Transform taking seed space to DTI space (either FLIRT matrix of FNIRT warpfield), per subject.

:Inverse XFM: Transform taking DTI space to seed space, per subject.

:Samples: Merged samples derived from bedpostX output.

Connectivity
------------
When connectivity is used as input rather than resting-state or difussion-weighted imaging data, the `masking` and
`connectivity` tasks are skipped. This means that connectivity matrices must be presented in the way *CBPtools*
generates them.

:Connectivity Matrix: A seed by target connectivity matrix. The number of seed voxels on the first dimension must match
   the number of seed voxels in the seed mask image. The order in which the seed voxels are listed along the first
   dimension depends on the order that was used to extract the voxels from the mask (i.e., F- or C-contiguous order).
   For *CBPtools* output, the 'dmri' `input_data_type` provides them in F-order, while the 'rsfmri' `input_data_type`
   provides them in C-order.

:Seed Indices: This is a 2-dimensional NumPy array (in .npy format) with shape seed voxels by 3. The order in which the
   indices appear should be identical to the order in which they appear in the connectivity matrix.

References
----------
Plachti A, Eickhoff SB, Hoffstaedter F, Patil KR, Laird AR, Fox PT, Amunts K, Genon S (2019): Multimodal parcellations
and extensive behavioral profiling tackling the hippocampus gradient. Cereb Cortex.

Pruim RHR, Mennes M, van Rooij D, Llera A, Buitelaar JK, Beckmann CF (2015): ICA-AROMA: A robust ICA-based strategy
for removing motion artifacts from fMRI data. Neuroimage 112:267–277.

Salimi-Khorshidi G, Douaud G, Beckmann CF, Glasser MF, Griffanti L, Smith SM (2014): Automatic denoising of functional
MRI data: combining independent component analysis and hierarchical fusion of classifiers. Neuroimage 90:449–468.