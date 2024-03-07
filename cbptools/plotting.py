from .image import make_hollow
from scipy.ndimage import find_objects
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster import hierarchy

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_scores(data, out_file, x, y=None, figure_format='png', hue=None,
                order=None, source=None, plot_type: str = 'boxplot') -> None:
    """Plot a box- or pointplot

    Parameters
    ----------
    data : pandas.DataFrame
        Pandas dataframe containing the data to be plotted.
    out_file : str
        Output file name for the figure
    x : str
        Name of the data column to be plotted on the x-axis
    y : str, optional
        Name of the data column to be plotted on the y-axis
    figure_format : str, optional, {'png', 'svg', 'pdf', 'ps', 'eps'}
        Format of the figures that will be saved to disk
    hue : str, optional
        Name of the data column to be plotted by color
    order : str, optional
        Order of display for the groups in y
    source : str, optional
        Name of the file where the data is taken from
    plot_type : str, optional
        The type of plot to be used. Allowed values are {'boxplot',
        'pointplot'}, with 'boxplot' being the default.
    """
    plt.ioff()
    sns.set_style('whitegrid',
                  {'font.family': 'Arial', 'font.sans-serif': 'Arial'})
    sns.set_context('paper', font_scale=1.)
    sns.set_palette('bright')
    fig = plt.figure(figsize=(3, 4), frameon=True, facecolor='w')
    ax = fig.add_subplot()

    if plot_type == 'boxplot':
        sns.boxplot(x=x, y=y, hue=hue, data=data, ax=ax, showfliers=False,
                    saturation=.6, width=.8, dodge=True, linewidth=.75,
                    notch=True, order=order)

    elif plot_type == 'pointplot':
        sns.pointplot(x=x, y=y, data=data, ax=ax)

    else:
        raise ValueError('Unknown plot type: \'%s\'' % plot_type)

    if ax:
        ax.set_ylabel(y.replace('_', ' ').title(), fontsize=10)
        ax.set_xlabel(x.replace('_', ' ').title(), fontsize=10)
        ax.tick_params(labelsize=8)

    if source:
        fig.text(.0, .0, 'Source: %s' % source, ha='left', fontsize=8)

    if hue:
        ax.legend(bbox_to_anchor=(.85, 1), loc=2, borderaxespad=0., title='k',
                  frameon=False)

    sns.despine(offset=10, trim=True)
    fig.tight_layout()
    fig.savefig(out_file, format=figure_format, bbox_inches='tight',
                pad_inches=0.01, facecolor=fig.get_facecolor(),
                transparent=False)


def plot_heatmap(data: np.ndarray, out_file: str, source: str = None,
                 plot_type: str = 'heatmap', **kwargs) -> None:
    """ Plot a volumetric ROI, color coding the cluster labels.

    Parameters
    ----------
    data : np.ndarray
        2D data matrix of pairwise similarity values
    out_file : str
        Output filename for the plotted figure
    source : str, optional
        Filepath of the source data
    plot_type : str, optional
        Type of plot to generate. Allowed values are {'clustermap', 'heatmap'},
        with 'heatmap' being the default.
    kwargs : dict, optional
        Keyword arguments passed to the seaborn heatmap function
    """
    if kwargs is None:
        kwargs = dict()

    plt.ioff()
    sns.set_style('whitegrid',
                  {'font.family': 'Arial', 'font.sans-serif': 'Arial'})
    sns.set_context('paper', font_scale=1.)
    sns.set_palette('bright')

    fig = plt.figure(figsize=(12, 8))

    if plot_type == 'clustermap':
        # method='average' vs. method='centroid'
        y = hierarchy.linkage(data, method='average')
        z = hierarchy.dendrogram(y, orientation='right', no_plot=True)
        index = z['leaves']
        data = data[index, :]
        data = data[:, index]

    if kwargs.get('xticklabels', None) is None:
        kwargs['xticklabels'] = False

    if kwargs.get('yticklabels', None) is None:
        kwargs['xticklabels'] = False

    if kwargs.get('robust', None) is None:
        kwargs['robust'] = True

    ax = sns.heatmap(
        data,
        cbar_kws=dict(
            use_gridspec=False,
            location="right",
            shrink=0.3,
            aspect=5,
            anchor=(-0.25, 1.0),
            ticks=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        ),
        **kwargs
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(axis='y', direction='in')

    if source:
        fig.text(.122, .09, 'Source: %s' % source, ha='left', fontsize=8)

    plt.savefig(out_file, format='png', bbox_inches='tight', pad_inches=0.01,
                facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig)


def plot_comparison(data: np.ndarray, out_file: str,
                    source: str = None, figure_format: str = 'png',
                    title: str = None) -> None:

    plt.tight_layout()
    plt.ioff()
    sns.set_style('whitegrid',
                  {'font.family': 'Arial', 'font.sans-serif': 'Arial'})
    sns.set_context('paper', font_scale=1.)
    sns.set_palette('bright')

    plt.figure(figsize=(2, 2))
    ax = sns.heatmap(data, annot=True, linewidth=.5, square=True, cbar=False)
    ax.set_ylabel('')
    ax.set_xlabel('group clusters')
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis=u'both', which=u'both', length=0)

    fig = ax.get_figure()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if title:
        ax.set_title(title, weight='bold')

    if source:
        fig.text(.1, .0, 'Source: %s' % source, ha='left', fontsize=8)

    fig.savefig(out_file, format=figure_format, bbox_inches='tight',
                pad_inches=0.01)
    plt.close(fig)


def plot_volumetric_roi(data: np.ndarray, out_file: str, view: str = 'orig',
                        skew: tuple = None, facecolor: str = 'bright',
                        edgecolor: str = 'dark', lrflip: bool = False) -> None:
    """ Plot a volumetric ROI, color coding the cluster labels.

    Parameters
    ----------
    data : np.ndarray
        3D data array of a NIfTI image
    out_file : str
        Output filename for the .svg figure
    view : str, optional
        The viewing angle of the ROI. Allowed values are {'orig',
        'left', 'right', 'superior', 'inferior', 'anterior',
        'posterior'}
    skew : tuple, optional
        The elevation and azimuth to rotate the ROI. If set to
        (0, 0) it is difficult to see the 3D effect. If left empty,
        default skewing elevation and azimuth values are used.
    facecolor : str, optional
        The colors the voxels will take, ordered by cluster-id. The
        color palette from sns.color_palette() is used. Allowed values
        are are {'deep', 'muted', 'pastel', 'bright', 'dark',
        'colorblind'}
    edgecolor : str, optional
        The colors the borders around the voxels will take, ordered
        by cluster-id. Same as facecolor.
    lrflip : bool, optional
        Left-right flip of the ROI for viewing purposes

    """

    # Modify data matrix for optimal viewing
    data = data[
        find_objects(data.astype(bool).astype(int))[0]]  # remove whitespace

    if lrflip:
        data = np.fliplr(data)

    data = make_hollow(data)  # Remove voxels that aren't visible

    # Viewing angle
    views = {'orig': (30, 320), 'left': (0, 0), 'right': (0, 180),
             'superior': (90, 90), 'inferior': (270, 90), 'anterior': (0, -90),
             'posterior': (0, -270)}

    if not skew:
        skew = (10, -10) if {'inferior',
                             'posterior'}.intersection({view}) else (10, 10)

    elev, azim = tuple(map(sum, zip(views[view], skew)))

    # Colors
    face_palette = sns.color_palette(facecolor).as_hex()
    edge_palette = sns.color_palette(edgecolor).as_hex()
    facecolors = np.empty(data.shape, dtype=object)
    edgecolors = np.empty(data.shape, dtype=object)

    for i, k in enumerate(np.unique(data)[1:]):
        facecolors[data == k] = face_palette[i]
        edgecolors[data == k] = edge_palette[i]

    # Plotting
    plt.ioff()
    fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
    fig.tight_layout()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev, azim)
    ax.voxels(data, facecolors=facecolors, edgecolors=edgecolors)
    dim = np.max(data.shape) / 2
    ax.set_xlim(right=dim * 2)
    ax.set_ylim(top=dim * 2)
    ax.set_zlim(top=dim * 2)
    plt.axis('off')
    fig.savefig(
        out_file,
        transparent=True,
        bbox_inches='tight',
        pad_inches=0
    )
    plt.close(fig)
