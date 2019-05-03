#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cbptools.utils import sort_files
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def main(participants: str, validity: list, internal_validity_metrics: list, output: dict, figure_format: str='png'):
    """ Generate a summary of the internal validity results.

    This script merges internal cluster validity into one table and generates a figure for a summary viewing.

    Parameters
    ----------
    participants : str
        Path to the participant information dataframe file (as .tsv)
    validity : list
        Paths to the various internal validity metric reports (as .tsv)
    internal_validity_metrics: list
        List of metrics that can be found in the validity metric reports
    output : dict
        Dictionary containing the various output file paths.
    figure_format : str, optional, {'png', 'svg', 'pdf', 'ps', 'eps'}
        Format of the figures that will be saved to disk
    """

    # Merge internal validity scores by metric into subject x k (n_clusters) tables
    validity = sort_files(participants, validity, pos=-1, sep='_', index_col='participant_id')
    data = pd.concat((pd.read_csv(f, sep='\t', index_col=False) for f in validity), ignore_index=True)
    data.to_csv(output.get('table'), sep='\t', index=False)

    # Generate validity metric figure
    plt.ioff()

    # Max column width = 3
    if len(internal_validity_metrics) > 3:
        ncols = 3
        nrows = int(np.floor(len(internal_validity_metrics)/3))+1
    else:
        ncols = len(internal_validity_metrics)
        nrows = 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

    for metric, ax in zip(internal_validity_metrics, axes.flat[0:]):
        data.rename(columns={'n_clusters': 'clusters'}, inplace=True)
        sns.boxplot(x='clusters', y=metric, data=data, ax=ax, showfliers=False)

    fig.suptitle('Internal Validity Scores')
    sns.despine(offset=10, trim=True)
    plt.savefig(output.get('figure'), format=figure_format)


if __name__ == '__main__':
    main(
        participants=snakemake.input.get('participants'),
        validity=snakemake.input.get('validity'),
        internal_validity_metrics=snakemake.params.get('internal_validity_metrics'),
        output={
            'table': snakemake.output.get('table'),
            'figure': snakemake.output.get('figure')
        },
        figure_format=snakemake.params.get('figure_format')
    )
