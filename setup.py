#!/usr/bin/env python

from setuptools import setup, find_packages
from os.path import join as opj, dirname
import re

version_file = 'cbptools/_version.py'
readme = opj(dirname(__file__), 'README.md')
mo = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", open(version_file, "rt").read(), re.M)

if mo:
    version = mo.group(1)
else:
    raise RuntimeError(f'Unable to find version string in {version_file}.')

try:
    import pypandoc
    long_description = pypandoc.convert(readme, 'rst')

except (ImportError, OSError) as exc:
    print('WARNING: pypandoc failed to import or threw an error while converting '
          f'README.md to RST: {exc} .md version will be used as is')
    long_description = open(readme).read()

setup(
    name='cbptools',
    version=version,
    description='Regional Connectivity-Based Parcellation tool for Python using Snakemake',
    long_description=long_description,
    url='https://github.com/nreuter/cbptools',
    author='Niels Reuter',
    author_email='niels.reuter@gmail.com',
    license='Apache',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'cbptools=cbptools.main:main',
        ],
    },
    install_requires=[
        'datrie@git+https://github.com/pytries/datrie.git',
        'matplotlib>=2.2.2',
        'nibabel>=2.2.1',
        'numpy>=1.14.5',
        'pandas>=0.23.2',
        'pyyaml>=3.12',
        'scikit-learn>=0.19.1',
        'scipy>=1.1.0',
        'seaborn>=0.8.1',
        'snakemake>=5.4.5'
    ],
    extras_require={
        'devel-docs': [
            # for converting README.md -> .rst for long description
            'pypandoc',
        ],
    },
)
