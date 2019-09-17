=================
Quick Start Guide
=================

Connectivity-Based Parcellation Tools
=====================================
Regional **connectivity-based parcellation** (rCBP) is a procedure for revealing the differentiation within a
region of interest (ROI) based on its long-range connectivity profiles. *CBPtools*
:cite:`reuter:submitted` is an open-source distributed workflow for applying this procedure using the
Python programming language.

The rCBP procedure groups ROI voxels based on similarity in their connection strengths to a set of target voxels,
i.e., their connectivity profile. The grouping is done using the
`k-means clustering <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ algorithm, which
is optimized such that within-group similarity of connectivity profiles is high and between-group similarity is low.
Grouped voxels form homogenous units, i.e., parcels, with regards to the measured connectivity marker within an ROI
that best describes the input data at hand. The rCBP procedure can map functional or structural clusters within a
particular ROI. Individual subject results are grouped by applying a
`hierarchical clustering <https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>`_
algorithm on the individual clustering results.

For a more detailed overview of the procedure, we recommend reading :cite:`eickhoff:2015` and :cite:`eickhoff:2018`.


.. toctree::
    :maxdepth: 2
    :caption: Contents:

.. toctree::
    :caption: Overview
    :name: overview
    :hidden:
    :maxdepth: 1

    overview/workflow
    overview/quickstart
    overview/examples

.. toctree::
    :caption: Getting Started
    :name: getting_started
    :hidden:
    :maxdepth: 1

    instructions/installation
    instructions/configuration
    instructions/input_data
    instructions/parameters
    instructions/validation_and_setup
    instructions/execution

.. toctree::
    :caption: Project Info
    :name: info
    :hidden:
    :maxdepth: 1

    zbibliography
    abbreviations
    license
