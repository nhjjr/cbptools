#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CBP tools is a NeuroImaging tool for Connectivity-Based Parcellation
-----------------------------------------------------------------------
The goal of cbptools is to handle neuroimaging data for the generation of brain maps. It interfaces with a variety of
other modules, such as nilearn and scikit-learn, to facilitate research on mapping the human brain. The library provides
input/output functions for data transformation, metrics for cluster validation, as well as visualization and reporting
tools tailored towards Connectivity-Based Parcellation (CBP).

Modules
-------
clean                   --- Signal cleaning utilities
cluster                 --- Utilities for cluster data
connectivity            --- Tools for computing connectivity profiles
image                   --- Utilities for handling and transforming neuroimaging data

"""
from . import clean, cluster, connectivity, exceptions, image, utils
from ._version import __version__

__readthedocs__ = 'http://docs.inm7.de'
__all__ = ['clean', 'cluster', 'connectivity', 'exceptions', 'image', 'utils', '__version__', '__readthedocs__']
