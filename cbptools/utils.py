#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for logging and file processing"""

from typing import Union
import pandas as pd
import math
import yaml
import numpy as np
import zipfile


def config_get(keymap, config, default=None):
    """Retrieve a nested value from a dict using a key mapping"""
    def delve(data: dict, *args: str) -> Union[str, dict, None]:
        if not isinstance(data, dict):
            return None

        return data.get(args[0], None) if len(args) == 1 \
            else delve(data.get(args[0], {}), *args[1:])

    mapping = keymap.split('.')
    value = delve(config, *mapping)
    return value if value is not None else default


class TColor:
    reset_all = "\033[0m"

    bold = "\033[1m"
    dim = "\033[2m"
    underlined = "\033[4m"
    blink = "\033[5m"
    reverse = "\033[7m"
    hidden = "\033[8m"

    reset_bold = "\033[21m"
    reset_dim = "\033[22m"
    reset_underlined = "\033[24m"
    reset_blink = "\033[25m"
    reset_reverse = "\033[27m"
    reset_hidden = "\033[28m"

    default = "\033[39m"
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    light_gray = "\033[37m"
    dark_gray = "\033[90m"
    light_red = "\033[91m"
    light_green = "\033[92m"
    light_yellow = "\033[93m"
    light_blue = "\033[94m"
    light_magenta = "\033[95m"
    light_cyan = "\033[96m"
    white = "\033[97m"

    bg_default = "\033[49m"
    bg_black = "\033[40m"
    bg_red = "\033[41m"
    bg_green = "\033[42m"
    bg_yellow = "\033[43m"
    bg_blue = "\033[44m"
    bg_magenta = "\033[45m"
    bg_cyan = "\033[46m"
    bg_light_gray = "\033[47m"
    bg_dark_gray = "\033[100m"
    bg_light_red = "\033[101m"
    bg_light_green = "\033[102m"
    bg_light_yellow = "\033[103m"
    bg_light_blue = "\033[104m"
    bg_lightmagenta = "\033[105m"
    bg_lightcyan = "\033[106m"
    bg_white = "\033[107m"


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


def npy_header(npy):
    """Takes a path to an .npy file. Generates a sequence of (shape, np.dtype).
    """
    with open(npy, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        return shape, dtype


def npz_headers(npz):
    """Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape, np.dtype).
    """
    with zipfile.ZipFile(npz) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue

            npy = archive.open(name)
            version = np.lib.format.read_magic(npy)
            shape, fortran, dtype = np.lib.format._read_array_header(npy,
                                                                     version)
            yield name[:-4], shape, dtype