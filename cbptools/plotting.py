from cbptools.image import make_hollow
from scipy.ndimage import find_objects
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_volumetric_roi(data: np.ndarray, out_file: str, view: str = 'orig',
                        skew: tuple = None, facecolor: str = 'bright',
                        edgecolor: str = 'dark') -> None:
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
    """

    # Modify data matrix for optimal viewing
    data = data[
        find_objects(data.astype(bool).astype(int))[0]]  # remove whitespace
    data = np.fliplr(data)  # flip to neurological view
    data = make_hollow(data)  # Remove voxels that aren't visible

    # Viewing angle
    views = {'orig': (30, 320), 'left': (0, 0), 'right': (0, 180),
             'superior': (90, 90), 'inferior': (270, 90), 'anterior': (0, -90),
             'posterior': (0, -270)}
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
    ax = fig.gca(projection='3d')
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
