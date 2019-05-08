#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for logging and file processing"""

import pandas as pd
import nibabel as nib
import math
import yaml

spatialimage = nib.spatialimages.SpatialImage


class TColor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class CallCountDecorator:
    """Counts number of times a method is called"""
    def __init__(self, method):
        self.method = method
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.method(*args, **kwargs)


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def sort_files(participants: str, files: list, pos: int = -1, sep: str = '_',
               index_col: str = 'participant_id') -> list:
    """
    Parameters
    ----------
    participants : str
        Path to a participants tsv file. The contents of the index
        column (default: 'participant_id') should match
        part of the filename. The ordering of the participants in
        this file will determine the ordering of the listed
        files.
    files : list
        List of filenames to be sorted based on the order of values
        of index_col in the participants file.
    pos : int
        Position at which the participant id found in the filename
        when splitting with the defined separator (sep)
    sep : str
        Separator used to split the filename into multiple parts.
    index_col : str
        The column in participants that defines the participant_id
        which is to be found in the list of filenames.

    Returns
    -------
    list
        Sorted input file names
    """

    df = pd.read_csv(participants, sep='\t').set_index(index_col)
    participant_id = df.index.values.astype(str).tolist()
    sorted_files = sorted(
        files,
        key=lambda x: participant_id.index(x.split(sep)[pos].split('.')[0])
    )
    return sorted_files


def readable_bytesize(size: int, itemsize: int = 1):
    size *= itemsize
    if size == 0:
        return "0B"

    units = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size, 1000)))
    s = round(size / math.pow(1000, i), 2)
    readable = '%s %s' % (s, units[i])

    return readable


def bytes_to(bytes: int, to: str, bsize: int = 1024):
    a = {'kb': 1, 'mb': 2, 'gb': 3, 'tb': 4, 'pb': 5, 'eb': 6}
    r = float(bytes)
    for i in range(a[to]):
        r = r / bsize

    return r


def pyyaml_ordereddict(dumper, data):
    value = []
    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)
        value.append((node_key, node_value))

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)
