Installation
============

Requirements
------------
CBPtools requires a Python3 (>=3.5) installation. All of its dependencies will be installed except for FSL's
`probtrackx2 <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide#PROBTRACKX_-_probabilistic_tracking_with_crossing_fibres>`_.
This tool is necessary to perform probabilistic tractography on diffusion-weighted imaging data. If you are only
interested in using resting-state fMRI time-series or connectivity matrices as input, `probtrackx2` is not necessary.

To see whether `probtrackx2` is installed and accessible in your environment, try the following terminal command:

.. code-block:: bash

    probtrackx2 --help

If it is not available, use `these instructions <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_ to install
FSL.

Installation instructions
-------------------------
It is recommended to use a dedicated virtual environment (see
`virtualenv documentation <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/>`_):

.. code-block:: bash

    python3 -m venv ~/.venv/cbptools
    source ~/.venv/cbptools/bin/activate

CBPtools can be installed using pip by running the following terminal command:

.. code-block:: bash

    pip install cbptools
