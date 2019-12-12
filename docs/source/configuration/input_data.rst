.. |br| raw:: html

    <br/>

.. raw:: html

    <style>
        .section #data-fields li { list-style: none; margin: 0xp; }
        .section #data-fields blockquote { margin: 0px; padding: 0px; }
        .section #data-fields p { margin: 0px; padding: 0px; }
        .title { font-weight: bold; color: green; }
        .desc { padding: 5px 0px 15px 25px; display: block; }
        .schema { color: black; font-weight: bold }
        .modality { color: red; font-weight: bold }
        .green { color: green; font-weight: bold }
    </style>

.. role:: title
.. role:: desc
.. role:: schema
.. role:: modality
.. role:: green

.. _ConfigurationInputData:

==========
Input Data
==========
The first step of configuring a regional CBP project is to define the `modality` of the input `data`. The files that
are required or allowed differ per modality. Currently supported modalities are resting-state functional MRI (rsfMRI)
and diffusion MRI (dMRI). It is also possible to provide connectivity matrices directly as input. This will skip the
steps in the workflow used to generate connectivity matrices from rsfMRI or dMRI data. All input data should be
quality controlled prior to using *CBPtools*, as only marginal validation is performed on the input data. Faulty data
may halt processing, but in a worst-case scenario such data may provide untrustworthy output.

Regardless of the chosen modality, the following input data can be provided: (1) a binary 3-dimensional ROI file in the
NIfTI image data format (:green:`masks:seed`), and (2) an optional 3-dimensional target mask in the same data
format, used to define the connections that are considered for each ROI voxel. If not provided by the user, a modified
FSL (http://www.fmrib.ox.ac.uk/fsl/) distributed average Montreal Neurological Institute (MNI) 152 T1 whole-brain gray
matter group template (2mm isotropic) will be used (:green:`masks:target`). In this case, the input data should
match the same MNI152 template as well. If connectivity matrices are entered directly, the target mask will not be
used. Lastly, (3) a participants file as a tab-separated text file with a column called 'participant_id' containing all
unique identifiers of hte subjects to be included in this project (:green:`participants`).

Resting-State Data
==================
A 4-dimensional time-series NIfTI image per subject (as defined in the participants file) is required
(:green:`time_series`), optionally accompanied by a tab-separated text file containing confounds for each time point as
columns (:green:`confounds`). *CBPtools* assumes that the rsfMRI data has been treated with necessary fMRI
preprocessing including realignment and normalisation to a template space. If the default target mask is used, then the
template space must be MNI152 with 2mm isotropic voxels. Denoising based on independent component analysis like
Automatic Removal of Motion Artifacts (ICA-AROMA) :cite:`pruim:2015` or FMRIB's ICA-based X-noiseifier (FIX)
:cite:`salimi:2014` is encouraged if suitable.

Diffusion-Weighted Imaging Data
===============================
*CBPtools* uses probabilistic diffusion tractography to generated connectivity matrices for the dMRI modality.
Therefore, input necessary to perform FSL's probabilistic diffusion tractography (PROBTRACKX2) is required per subject,
consisting of: (1) Outputs from Bayesian Estimation of Diffusion Parameters Obtained using Sampling Techniques
(BEDPOSTX) (:green:`samples`), (2) a brain extraction (BET) binary mask file (:green:`bet_binary_mask`), (3) a
transform file taking seed space to DTI space (either a FLIR matrix or FNIR warpfield) (:green:`xfm`), and (4) a file
describing the transformation from DTI space to seed space (:green:`inv_xfm`). Each of these files is subject-specific
and can be obtained from FSL's BEDPOSTX output.

Connectivity Data
=================
Connectivity matrices may be provided as source input in lieu of rsfMRI or dMRI data. The `masking` and `connectivity`
tasks are then skipped. This means that connectivity matrices must be presented in the way *CBPtools* generates them.
They must be provided in an ROI-voxel by target-voxel shape as a NumPy array. This can either be done in the
uncompressed .npy format, or as a compressed .npz file. In case of the latter, make sure that the array is stored as
`connectivity.npy` inside of the .npz archive (:green:`connectivity`). Along with the connectivity matrix, a binary
3-dimensional mask of the ROI in NIfTI image data format is expected. The number of voxels in this mask must coincide
with the number of seed voxels on the first dimension of the connectivity matrix (:green:`masks:seed`). Lastly, a NumPy
array (.npy) of seed voxel coordinates must be provided in the order the voxels are represented in the connectivity
matrix. This is used to map the clustering results onto the ROI mask for visualization purposes
(:green:`seed_coordinates`).

Data fields
===========

.. jinja:: schema

    {% for k, v in data.items() %}
    {% if v.type is not defined %}

       * :title:`{{k}}:`

       {% for k1, v1 in v.items() %}
       {% if v1.type is not defined %}

           * :title:`{{k1}}:`

           {% for k2, v2 in v1.items() %}
           {% if v2.type is not defined %}

               * :title:`{{k2}}:`

               {% for k3, v3 in v2.items() %}

                   * :title:`{{k3}}:` :schema:`{{v3.type}}{% if v3.required %}, required{% endif %}{% if v3.allowed %}, allowed = {{v3.allowed}}{% endif %}, modality =`
                     :modality:`{% if v3.dependency and v3.dependency[0].modality %}{{v3.dependency[0].modality}}{% else %}any{% endif %}`
                     :desc:`{% if v3.desc %}{{v3.desc}}{% else %}No description given{% endif %}`

               {% endfor %}
           {% else %}

               * :title:`{{k2}}:` :schema:`{{v2.type}}{% if v2.required %}, required{% endif %}{% if v2.allowed %}, allowed = {{v2.allowed}}{% endif %}, modality =`
                 :modality:`{% if v2.dependency and v2.dependency[0].modality %}{{v2.dependency[0].modality}}{% else %}any{% endif %}`
                 :desc:`{% if v2.desc %}{{v2.desc}}{% else %}No description given{% endif %}`

           {% endif %}
           {% endfor %}
       {% else %}

           * :title:`{{k1}}:` :schema:`{{v1.type}}{% if v1.required %}, required{% endif %}{% if v1.allowed %}, allowed = {{v1.allowed}}{% endif %}, modality =`
             :modality:`{% if v1.dependency and v1.dependency[0].modality %}{{v1.dependency[0].modality}}{% else %}any{% endif %}`
             :desc:`{% if v1.desc %}{{v1.desc}}{% else %}No description given{% endif %}`

       {% endif %}
       {% endfor %}
    {% else %}

       * :title:`{{k}}:` :schema:`{{v.type}}{% if v.required %}, required{% endif %}{% if v.allowed %}, allowed = {{v.allowed}}{% endif %}, modality =`
         :modality:`{% if v.dependency and v.dependency[0].modality %}{{v.dependency[0].modality}}{% else %}any{% endif %}`
         :desc:`{% if v.desc %}{{v.desc}}{% else %}No description given{% endif %}`

    {% endif %}
    {% endfor %}
