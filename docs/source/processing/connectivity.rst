.. _TaskConnectivity:

============
Connectivity
============
The connectivity task differs between the rsfMRI and dMRI modalities. This is the only part of the workflow where both
modalities differ, as from the clustering onwards all output is treated the same.

Resting-State fMRI
==================
This task will be executed in parallel (if possible), processing multiple subjects at the same time. It is recommended
to set the `--resources io=x` parameter when executing Snakemake to reduce the amount of connectivity jobs that can be
run in parallel if file system latency is an issue.

The time-series of a subject will be loaded and optionally smoothing is applied using the
nibabel.processing.smooth_image function. Next, the seed- and target voxels are taken from the time-series using the
seed- and target masks, respectively. Nuisance signal regression is then optionally applied to the time-series of the
seed- and target voxels, followed by optional band-pass filtering. Band-pass filtering is recommended, but is not a
default setting as the repetition time must manually be entered in the configuration file.

At this point *CBPtools* searchers for low-variance voxels both in the seed and target. If the number of low-variance
voxels exceeds the predefined threshold, then an empty connectivity matrix is instead given as output. This allows the
workflow to continue processing other subjects, and processing is only halted prior to the grouping tasks. Subjects can
then easily be removed or corrected, after which the workflow can be resumed. Note that already processed files do not
need to be processed again.

The correlation between the seed-voxel time-series and target-voxel time-series is now computed and optionally
arctanh transformed (using np.arctanh). Lastly, if the option is set in the configuration file, the principal
component analysis is performed on the connectivity matrix. This helps to reduce the number of features (i.e., the
target voxels) to a lower number of components and may speed up subsequent clustering.

The resulting connectivity matrix is stored either as an uncompressed .npy file, or as a compressed .npz file with the
array being named connectivity.npy.


Diffusion MRI
=============
As for resting-state fMRI, this task will also be executed in parallel (if possible), running multiple instances of
PROBTRACKX2. The `omatrix2` file (called fdt_matrix2.dot) is then densified and optionally cubic transformed
(np.power(connectivity, 1 / 3)). As with the rsfMRI connectivity, principal component analysis can optionally be
applied. Since FSL uses F-contiguous ordering, the order of the seed voxels in the connectivity matrix is reordered as
if C-contiguous ordering were used. This is only performed on the seed voxels, and the target voxels remain in F-order.
For further processing with *CBPtools* their order does not matter.

The resulting connectivity matrix is stored either as an uncompressed .npy file, or as a compressed .npz file with the
array being named connectivity.npy.

Running PROBTRACKX2 and densifying its output are considered two separate tasks in the workflow, although this has no
further bearing on the results.