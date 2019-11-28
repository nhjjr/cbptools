#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CBPtools is a NeuroImaging tool for Connectivity-Based Parcellation"""
from . import (clean, cluster, connectivity, exceptions, image, utils,
               plotting, workflow)
from ._version import __version__

__readthedocs__ = 'https://cbptools.readthedocs.io/'
__all__ = [
    'clean',
    'cluster',
    'connectivity',
    'exceptions',
    'image',
    'utils',
    'plotting',
    'workflow',
    '__version__',
    '__readthedocs__'
]
