# Connectivity-Based Parcellation Tools
Python implementation of the Connectivity-Based Parcellation toolbox used by INM-7. This package offers a workflow for 
regional multi-modal connectivity-based parcellation (CBP). Currently implemented modalities are resting-state 
functional connectivity (`'fmri'`) and diffusion-weighted imaging (`'dmri'`). Subject-wise imaging data is transformed 
into region-of-interest (ROI)-based connectivity matrices (ROI to whole-brain), of which clusters are derived using the 
[k-means clustering](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) algorithm. Various
validity metrics are then computed for the resulting cluster labels, followed by a 
[hierarchical clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) of all subject-wise labels
into a group-level clustering.

# Getting Started
These instructions will assist in installing cbptools and its dependencies and provide a manual for reproducing the
results from the [publication](http://not-yet-submitted).

## Prerequisites
cbptools requires a Python (3.6+) installation and will install all the python packages it needs except for FSL's 
[probtrackx2](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide#PROBTRACKX_-_probabilistic_tracking_with_crossing_fibres), 
which is necessary to compute the connectivity matrices for applying CBP on the dmri modality for diffusion-weighted 
imaging (DWI). If you are only interested in resting-state CBP, `probtrackx2` is not necessary.

To see whether `probtrackx2` is installed and accessible in your environment, try the following terminal command:

    probtrackx2 --help

If it is not available, use [these instructions](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) to install FSL.

## Installing
We recommend installing cbptools and its dependencies in a separate virtual environment. To install, run the following
command in the terminal:

    pip install git+ssh://git@ime263.ime.kfa-juelich.de/niels/cbptools.git

To quickly get started, download one of the example files from the `examples/` folder. Here, `setup_insula_fmri.py` and
`setup_insula_dmri.py` will set up a regional CBP project reproducing the fMRI and dMRI results from the 
[publication](http://not-submitted-yet) respectively. Open the example file with a text editor and change the file 
paths to where the data is located. For example, in `setup_insula_fmri.py` the following lines need to be edited to 
reflect the proper paths:

```python
from cbptools import Project
study = Project()
study.modality = 'fmri'
study.work_dir = '/path/to/work_dir'
study.set_participants([
    's001', 's002', 's003', 's004', 's005', 's006', 's007', 's008', 's009', 's010', 's011', 's012', 's013', 's014',
    's015', 's016', 's017', 's018', 's019', 's020'
])
study.add_mask('seed', '/path/to/data/HVOX_R_Insula_2mm_thr50.nii.gz')
study.add_path('dataset', '/path/to/data/dataset')
study.add_file('time_series', '{participant_id}/time_series.nii')
study.add_file('confounds', '{participant_id}/confounds.tsv', sep='\t')
```

Note that `study.work_dir` defines the location in which the output will be saved, and the `dataset` path (in this case 
'/path/to/data/dataset') will be prepended to all files belonging to the dataset. Once you have finished editing and 
saved your changes, run the following command from within the folder the `setup_insula_fmri.py` file is located:

    python setup_insula_fmri.py

This will set up the project and save all necessary files in the project folder. Next, change directory to the work_dir:

    cd /path/to/work_dir
    
From here, initiate the snakemake [1] workflow manager. The command below starts the workflow with 8 threads and 
limiting memory resources to 20GB. Change this to match your system's capabilities.

    snakemake -j 8 --resources mem_mb=20000

The pipeline is now running and once complete, the summary results can be found in the summary directory within the
work_dir.

# Instructions
## Input Data
The first step is to collect your input data. Define your `region-of-interest` as a binary NIfTI image in the same 
space as the time-series from your dataset. Commonly this is in MNI152 2mm space. Using nibabel, yo can check whether
your images meet the criteria (although there are alternative ways):

```python
import nibabel as nib
img = nib.load('path/to/region_of_interest.nii')
print(img.affine)
print(img.shape)
```

For the aforementioned MNI152 space, the affine should match the following:

    [[  -2.    0.    0.   90.]
     [   0.    2.    0. -126.]
     [   0.    0.    2.  -72.]
     [   0.    0.    0.    1.]]

And the shape should be 
    
    (91, 109, 91)

**Importantly**: make sure the NIfTI images are unzipped (e.g., .nii instead of .nii.gz), because especially with
large files (such as the time-series) the data cannot be memory mapped and loading them will cause significant
slowdown and higher memory usage than necessary.

Next, either a `participants.tsv` file or a list of participant ID's (as string) is needed to define the 
`{participant_id}` part of all dataset files (this part gets automatically replaced with the participant ID's during
execution of the workflow). A file with 5 subjects can look like:

participant_id | gender | age
--- | --- | --- |
571144 | M | 25
517239 | M | 23
812746 | M | 22
213421 | F | 26
204622 | F | 30

Take note of the separator (typically a comma (,) or semicolon (;) for .csv files, or a tab (\t) for .tsv files) as
it needs to be entered during setup for the file to be properly read. The participant_id column will be used to draw
participant data from the dataset. The remaining columns are unused.

The following input data depends on the selected modality and should be preprocessed (e.g., FIX-denoised, segmented,
normalized, etc.).

### Resting-State Dataset
<dl>
  <dt>Signal timeseries</dt>
  <dd>
    The time-series must be in a 4D NIfTI-image format (x, y, z, timepoints) for each subject.
  </dd>
  <dt>Confounds timeseries (optional)</dt>
  <dd>
    A .tsv file with signals as columns and a 1-line header. With the header it is possible to selected columns to
    use for nuisance signal regression.
  </dd>
</dl>

### Diffusion-Weighted Imaging
<dl>
  <dt>BET binary mask</dt>
  <dd>
    Each subject must have a <a href="https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#bet">BET</a> binary mask 
    file.
  </dd>
  
  <dt>XFM</dt>
  <dd>
    Transform taking seed space to DTI space (either FLIRT matrix or FNIRT warpfield), per subject
  </dd>

  <dt>Inverse XFM</dt>
  <dd>
    Transform taking DTI space to seed space, per subject
  </dd>

  <dt>Samples</dt>
  <dd>
    Merged samples derived from bedpostX output
  </dd>
</dl>

## Creating a cbptools project
Once all input data is ready a cbptools project must be created by specifying project parameters and files. Look at
`examples/setup_insula_fmri.py` or `examples/setup_insula_dmri.py` for a resting-state and diffusion-weighted imaging 
example respectively.

### Project Parameters
Project parameters are preceded by the Project() object. For instance, in the example given below the variable
`study` contains an instance of the Project() object.

```python
from cbptools import Project    
study = Project()
study.work_dir = 'path/to/file'
study.add_param('masking', 'threshold', 0.0)
```

To add a project parameter, use the table below and prefix the parameter by the variable containing the Project()
object (see `study.work_dir = 'path/to/file'`).

Parameter | Description
------------ | -------------
modality | **str, {'func', 'dwi'} :** The modality that should be assessed. There is a different process for deriving connectivity between the resting-state ('func') and diffusion-weighted imaging ('dwi') modalities
work_dir | **str :** Full path to the working directory in which the workflow should be created. The Snakemake workflow manager is then executed from within this directory
force_overwrite | **bool, optional :** If True, any existing project files will be overwritten. Note that this does not overwrite pipeline output

### Project Methods
The following methods are used to create a project:

Method | Description
------------ | -------------
set_participants(participants, sep, index_col) | Attach a participants file to the project. The participants argument is the path to the file, the sep argument defines what separator was used (e.g., '\t' for .tsv, ';' or ',' for .csv), and index_col indicates the name of the column used to store subject identifiers. These identifiers are used for selecting subject-specific data, and should therefore also be present in the dataset folder and/or filenames for these subjects. Alternatively, a list of strings containing participant ID's can be added instead of a table file, in which case the `sep` and `index_col` arguments are ignored.
add_param(task, parameter, value) | Adds a parameter to the project, where 'task' and 'parameter' make up the unique alias of the parameter. For a full list of parameters, see below.
add_parameter(task, parameter, value) | Identical to add_param.
add_mask(alias, filename) | Adds a mask image to the project, where alias is either {'seed', 'target', 'highres_template', 'lowres_template'}
add_file(alias, filename) | Adds a file to the project, where alias defines what role the file plays in the project, and filename is the full path to the file
add_path(alias, filename) | Identical to add_file, except more semantically correct when adding folders rather than files.
save() | Once all parameters and files have been added, this validates and saves the project files to the specified work_dir

When using add_file, all files are associated to the dataset. Therefore, each file must contain the wildcard
`{participant_id}` (with the exception of the 'dataset' path), which will be replaced by the actual participant ID's 
upon execution of the workflow. This indicates that the file exists once for each individual participant. Example:

```python
from cbptools import Project    
study = Project()

study.set_participants(
    participants='/path/to/file/participants.tsv',
    sep='\t',
    index_col='participant_id'
)
study.add_path('dataset', '/path/to/dataset')
study.add_file('time_series', '{participant_id}/time_series.nii')
```

Assuming we have 2 subjects with ID's 's001' and 's002' in `/path/to/file/participants.tsv`, the time_series files are:

    '/path/to/dataset/s001/time_series.nii'
    '/path/to/dataset/s002/time_series.nii'

### File list
This is a list of all files that can be added using the `Project.add_file()` or `Project.add_path()` method. All files 
not marked (optional) are required for the selected modality.

Alias | Modality | Description
------------ | ------------- | -------------
**dataset** | all | path to the dataset folder (this will be prepended to all filenames added using `add_file()`)
**bet_binary_mask** | dmri | (probtrackx2) bet binary mask file in diffusion space
**xfm** | dmri | (probtrackx2) Transform taking seed space to DTI space (either FLIRT matrix or FNIRT warpfield)
**inv_xfm** | dmri | (probtrackx2) Transform taking DTI space to seed space (compulsory when using a warpfield for seeds_to_dti)
**samples** | dmri | (probtrackx2) Basename for samples files - e.g. 'merged'
**time_series** | fmri | fMRI time-series file. Note that if these are .nii.gz instead of .nii the processing will be much slower
**confounds** | fmri | (optional) Confounds timeseries in table format with a header. Used columns are specified in the parameters section

### Parameter list
This is a list of all parameters that can be added using the 
`Project.add_param('taskname', 'parametername', 'value')` method. When a parameter is referred to in the log file, 
its identifier is "taskname.parametername" (e.g., 'masking.threshold').

All parameters marked (optional) or having a default value are not required. Note that when not specified, default
values will be used and a 'WARNING' level log entry is added.

Task | Parameter | type=(default) | Modality | Description
------------ | ------------ | ------------- | ------------- | -------------
**masking** | **threshold** | float=0.0 | all | threshold above which voxels in the ROI mask image are defined as 1's. This is only applied if the mask is not binary.
**masking** | **median_filter** | bool=False | all | Apply median filtering to the ROI mask.
**masking** | **median_filter_dist** | int=1 | all | Median filtering distance.
**masking** | **del_seed_from_target** | bool=False | fmri | Remove the ROI voxels from the whole-brain target mask
**masking** | **del_seed_expand** | int=0 | fmri | Expand the border around the ROI (in mm) for removal from the whole-brain target mask. This should only be applied if the input time-series data is smoothed
**masking** | **subsample** | bool=False | fmri | Apply subsampling to the whole-brain target mask to improve computational efficiency at minimal loss of specificity
**masking** | **resample_to_mni** | bool=True | all | Resample the input mask (both seed and target) to match the aforementioned MNI152 shape and affine
**masking** | **upsample_seed_to** | list | dmri | (optional) Upsample the seed mask to the specified voxel size (e.g., from [3, 3, 3] as 3mm isotropic to [1, 1, 1] as 1mm isotropic)
**masking** | **downsample_target_to** | list | dmri | (optional) Downsample the target mask to the specified voxel size, similar to how upsample_seed_to works
**connectivity** | **seed_low_variance_threshold** | float=0.05 | fmri | When more than this specified percentage of voxels within the seed has low or no variance over the entire time course, the processing will halt until the subject has been manually removed
**connectivity** | **target_low_variance_threshold** | float=0.1 | fmri | Same as seed_low_variance_threshold, except concerning voxels within the target masked area.
**connectivity** | **high_pass** | float | fmri | (optional) High-pass value for the band-pass filter. If this is used, low_pass and tr must also be specified
**connectivity** | **low_pass** | float | fmri | (optional) Low-pass value for the band-pass filter. If this is used, high_pass and tr must also be specified
**connectivity** | **tr** | float | fmri | (optional) Repetition time (in seconds) required for band-pass filtering
**connectivity** | **smoothing_fwhm** | int | fmri | (optional) FWHM kernel value for smoothing. If left empty, smoothing is skipped
**connectivity** | **confounds** | list | fmri | (optional) List of the columns in the confounds file that will be used for nuisance signal regression. If left empty when a confounds file is specified, all confounds will be used.
**connectivity** | **arctanh_transform** | bool=True | fmri | Arctanh transform applied to the connectivity matrix
**connectivity** | **pca_transform** | float | all | (optional) PCA transform applied to the connectivity matrix. This value is equivalent to n_components in sklearn.decomposition.PCA
**connectivity** | **dist_thresh** | float=5.0 | dmri | (probtrackx2) Discards samples shorter than this threshold (in mm)
**connectivity** | **loop_check** | bool=True | dmri | (probtrackx2) Perform loopchecks on paths - slower, but allows lower curvature threshold
**connectivity** | **c_thresh** | float=0.2 | dmri | (probtrackx2) Curvature threshold
**connectivity** | **step_length** | float=0.5 | dmri | (probtrackx2) Steplength in mm
**connectivity** | **n_samples** | int=200 | dmri | (probtrackx2) Number of samples
**connectivity** | **n_steps** | int=2000 | dmri | (probtrackx2) Number of steps per sample
**connectivity** | **correct_path_distribution** | bool=True | dmri | (probtrackx2) Correct path distribution for the length of the pathways
**connectivity** | **wait_for_file** | int=240 | dmri | Time in seconds the pipeline waits for files to appear on disk. Increase this when you expect high file system latency
**connectivity** | **cleanup_fsl** | bool=True | dmri | Remove all files probtrackx2 creates after the connectivity matrix has been extracted
**connectivity** | **cubic_transform** | bool=True | dmri | Apply a cubic transformation on the connectivity matrix
**clustering** | **n_clusters** | list | all | A list of cluster numbers to be evaluated (entered as [2, 3, 8] to receive a 2, 3, and 8-cluster solution)
**clustering** | **algorithm** | str='auto' | all | (sklearn.cluster.KMeans) K-means algorithm to use
**clustering** | **init** | str='random' | all | (sklearn.cluster.KMeans) Method for initialization
**clustering** | **max_iter** | int=10000 | all | (sklearn.cluster.KMeans) Maximum number of iterations of the k-means algorithm for a single run.
**clustering** | **n_init** | int=100 | all | (sklearn.cluster.KMeans) Number of time the k-means algorithm will be run with different centroid seeds
**clustering** | **linkage** | str='complete' | all | (scipy.cluster.hierarchy.linkage) The linkage algorithm to use (allowed values: 'single', 'average', 'complete')
**clustering** | **group_method** | str='agglomerative' | all | Method for obtaining group-level clustering results (allowed values: 'agglomerative', 'mode')
**clustering** | **internal_validity_metrics** | list | all | (optional) List of internal validity metrics to assess (allowed values: 'silhouette', 'davies-bouldin', 'calinski-harabasz', 'weak deletion stability')
**clustering** | **similarity_metric** | str='adjusted rand index' | all | Similarity metric to use to generate between-subject cluster comparisons and subject to group-level cluster comparisons (allowed values: 'adjusted rand index', 'adjusted mutual information', 'v measure')
**summary** | **figure_format** | str='png' | all | Format of the output figures generated for the summary (allowed values: 'png', 'svg', 'pdf', 'ps', 'eps')

References to external documentation:
1. [probtrackx2](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide#PROBTRACKX_-_probabilistic_tracking_with_crossing_fibres)
2. [sklearn.cluster.KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
3. [scipy.cluster.hierarchy.linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)

## Validation & Saving
Once the `Project.save()` method is called, all input files and parameters will be validated to see whether they meet
the requirements. Only if validation passes without errors (critical mistakes in the setup) will the project be
created. If there are errors, the log file is accessible from the current working directory and they must be addressed
before trying to create the project again.

Once validation passes, the mask files are processed using the input parameters for the `masking` task. The resulting
masks files are saved to the work_dir. Once completed, the working directory will have the following files:

* **cluster.json** : used by snakemake to define cluster parameters for each rule. When using a cluster (e.g., 
through slurm or qsub) this file defines timing, account name, cluster name, etc.
* **participants.tsv** : Input participants file converted to .tsv format. The index column is renamed to 
'participant_id'
* **project.yaml** : Contains all project setup parameters. This will be loaded when initiating the pipeline through
snakemake.
* **log/project.log** : All logged messages occurring during setup.
* **seed_mask.nii** : Binary ROI mask after processing
* **target_mask.nii** : Binary target mask after processing (default is MNI152 gray matter mask; fmri only)
* **highres_seed_mask.nii** : ROI mask stretched to a higher resolution matching the input highres template (dmri only)
* **Snakefile** : The workflow for the processing pipeline, used by snakemake to initiate it
* **scripts/** : Folder containing all scripts used by the workflow

## Running the pipeline
Once the project has been successfully created, change the directory to the working directory. From there, a 
snakemake command can be used in the terminal to start processing the data.

For a simple local run using 8 threads and at most 20GB of memory, try the command:

    snakemake -j 8 --resources mem_mb=20000

If you want to execute the pipeline on a cluster using SLURM, make sure to first edit the cluster.json file so that
all SLURM parameters are set properly.

    snakemake -j 999 -w 240 -u cluster.json --resources mem_mb=20000 -c "sbatch -p {cluster.partition} -n {cluster.n} -N {cluster.N} -t {cluster.time} -c {cluster.c} --mem-per-cpu={cluster.mem} --out={cluster.out} --job-name={cluster.name}"

Here, `-c "sbatch ..."` defines the command snakemake uses to start jobs through SLURM's sbatch. The `-w 240` is
added to ensure that snakemake waits 240 seconds for files to appear on the file system. File system latency may
cause snakemake to assume something went wrong with generating the file and we do not want that.

The snakemake tool is very flexible and for more information on how to use it, read the 
[snakemake documentation](https://snakemake.readthedocs.io/en/stable/index.html).

# Workflow
In this section the processing of the pipeline is described through its various tasks.

### (1) Connectivity (resting-state)
1. Load timeseries, roi_mask, target_mask (define target mask here)
2. Apply smoothing (if selected) using nibabel.processing.smooth_image on the timeseries
3. Apply the roi and target masks separately to teh timeseries
4. Nuisance signal regression (confound timeseries) of selected columns (optional) on roi and target-masked timeseries respectively
5. Calculate the correlation (seed_based_correlation, explain the function)
6. Apply arctanh transform (optional)
7. Apply PCA transform (optional) = reduces features (connectivity to target-voxels) of roi voxels

### (1) Connectivity (diffusion-weighted imaging)
1. Execute probtrackx2 to get omatrix2
2. Get fdt_matrix2.dot and create a dense matrix out of it.
3. Apply cubic transform (optional) = np.power(connectivity, 1 / 3)
4. Apply PCA transform (optional)

### (2) Participant-level Clustering
Load the connectivity matrix and apply kmeans (sklearn.cluster.KMeans) for k=n_clusters parameter.

### (3) Group-level clustering
1. Aggregate all single-participant clustering results into one matrix, sorted by the participant_id order in the
   participants file.
2. calculate pairwise hamming distance (y = pdist(x, metric='hamming')) on teh single-participant clustering matrix
3. Perform hierarchical clustering on y (z = hierarchy.linkage(y, method=linkage, metric='hamming')) with the
   requested linkage algorithm
4. Calculate the cophenetic correlation between z and y (hierarchy.cophenet(z, y))
5. Cut the tree at the requested cluster number: hierarchy.cut_tree(z, n_clusters=len(np.unique(x)))
6. Use the hierarchical clustering as a reference to relabel individual participant clustering results
7. If the method is agglomerative, use the hierarchical (agglomerative) clustering results as a group result
8. If the method is mode, apply np.mode on the relabeled individual participant clustering results and use that as
   a group result
9. Project the group clustering result onto the roi_mask for viewing the clustering results

### (4) Internal Validity
Compute the requested validity metrics per participant using the connectivity matrix as a feature array and the 
predicted labels for the participant. Note that for silhouette, the metric for calculating distance between instances
in the feature array is 'euclidean'.

The results for all participants are then merged into a table and used to generate boxplots for each validity metric
and each requested cluster number.

### (5) Individual Similarity
Use the requested similarity metric to calculate the pairwise similarity between all participant cluster labels for
each requested cluster number, resulting in a similarity matrix. This matrix is then plotted as a heatmap 
(unordered).

The same similarity matrix is then ordered using the leaves of a dendrogram of hierarchical clustering of the 
similarity matrix. This is then again plotted as a heatmap (ordered).

    y = hierarchy.linkage(similarity_matrix, method='centroid')
    z = hierarchy.dendrogram(y, orientation='right')

### (6) Group Similarity
Use the requested similarity metric to calculate the similarity between each participant's cluster labels and the
group cluster labels (per requested cluster number).

The resulting similarity data is then plotted as a boxplot.

The relabeling accuracy from step 3 is plotted as a boxplot.

The cophenetic correlation for each group-level clustering is now also taken and plotted as a pointplot.

# References
[1] Paper by Simon

[2] Paper by Sarah

[3] Snakemake ?

[4] FSL ?