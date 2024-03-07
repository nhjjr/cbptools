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
    python_requires='>=3.11.0',
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
        'matplotlib',  # current: 3.8.3
        'nibabel',  # current: 5.2.1
        'numpy',  # current: 1.26.4
        'pandas',  # current: 2.2.1
        'pyyaml',  # current: 6.0.1
        'scikit-learn',  # current: 1.4.1
        'scipy',  # current: 1.12.0
        'seaborn',  # current: 0.13.2
        'snakemake'  # current: 8.5.3
    ],
    extras_require={
        'devel-docs': [
            # for converting README.md -> .rst for long description
            'pypandoc',
        ],
    },
)
