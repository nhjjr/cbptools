#!/usr/bin/env python

from setuptools import setup, find_packages
from os.path import join as opj, dirname
import re

version_file = 'cbptools/_version.py'
readme = opj(dirname(__file__), 'README.md')
mo = re.search(
    r"^__version__ = ['\"]([^'\"]*)['\"]",
    open(version_file, "rt").read(),
    re.M
)

if mo:
    version = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in %s.' % version_file)

try:
    import pypandoc
    long_description = pypandoc.convert(readme, 'rst')

except (ImportError, OSError) as exc:
    print('WARNING: pypandoc failed to import or threw an error while '
          'converting README.md to RST: %s .md version will be used as is'
          % exc)
    long_description = open(readme).read()

setup(
    name='cbptools',
    version=version,
    python_requires='>=3.5.0',
    description='Regional Connectivity-Based Parcellation tool for Python',
    long_description=long_description,
    url='https://github.com/inm7/cbptools',
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
        # 'datrie@git+https://github.com/pytries/datrie.git',
        'matplotlib>=3.0.3',
        'nibabel>=2.5.0',
        'numpy>=1.17.0',
        'pandas>=0.25.0',
        'pyyaml>=5.1.1',
        'scikit-learn>=0.21.3',
        'scipy>=1.3.0',
        'seaborn>=0.9.0',
        'snakemake>=5.5.4'
    ],
    extras_require={
        'devel-docs': [
            # for converting README.md -> .rst for long description
            'pypandoc',
        ],
    },
)
