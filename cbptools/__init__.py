#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CBP tools is a NeuroImaging tool for Connectivity-Based Parcellation"""
from . import clean, cluster, connectivity, exceptions, image, utils, plotting
from ._version import __version__

__readthedocs__ = 'http://docs.inm7.de'
__all__ = ['clean', 'cluster', 'connectivity', 'exceptions', 'image', 'utils',
           'plotting', '__version__', '__readthedocs__']
