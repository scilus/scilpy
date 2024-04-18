# -*- coding: utf-8 -*-

from fury.colormap import distinguishable_colormap
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from scilpy.viz.backends.vtk import get_color_by_name, lut_from_colors


def convert_color_names_to_rgb(names):
    """
    Convert a list of VTK color names to RGB

    Parameters
    ----------
    names : list
        List of VTK color names.

    Returns
    -------
    colors : list
        List of RGB vtkColor.
    """

    return [get_color_by_name(name) for name in names]


BASE_10_COLORS = convert_color_names_to_rgb(["Blue",
                                             "Yellow",
                                             "Purple",
                                             "Green",
                                             "Orange",
                                             "White",
                                             "Brown",
                                             "Grey"])


def generate_n_colors(n, generator=distinguishable_colormap,
                      pick_from_base10=True, shuffle=False):
    """
    Generate a set of N colors. When using the default parameters, colors will
    always be unique. When using a custom generator, ensure it generates unique
    colors, excluding the ones listed in BASE_10_COLORS, if unicity is desired.

    Parameters
    ----------
    n : int
        Number of colors to generate.
    generator : function
        Color generating function f(n, exclude=[...]) -> [color, color, ...],
        accepting an optional list of colors to exclude from the generation.
    pick_from_base10 : bool
        When True, start picking from the base 10 colors before using
        the generator funtion (see BASE_COLORS_10).
    shuffle : bool
        Shuffle the color list before returning.

    Returns
    -------
    colors : np.ndarray
        A list of Nx3 RGB colors
    """

    _colors = []

    if pick_from_base10:
        _colors = np.array(BASE_10_COLORS[:min(n, 10)])

    if n - len(_colors):
        _colors = np.concatenate(
            (_colors, generator(n - len(_colors), exclude=_colors)), axis=0)

    if shuffle:
        np.random.shuffle(_colors)

    return _colors


def get_lookup_table(name):
    """
    Get a matplotlib lookup table (colormap) from a name or create
    a lookup table (colormap) from a list of named colors.

    Parameters
    ----------
    name : str
        Name of the lookup table (colormap) or a list of named colors
        (separated by a -) to form a lookup table (colormap) from.

    Returns
    -------
    matplotlib.colors.Colormap
        The lookup table (colormap)
    """

    if '-' in name:
        name_list = name.split('-')
        colors_list = [colors.to_rgba(color)[0:3] for color in name_list]
        cmap = colors.LinearSegmentedColormap.from_list('CustomCmap',
                                                        colors_list)
        return cmap

    return plt.colormaps.get_cmap(name)


def lut_from_matplotlib_name(name, value_range, n_samples=256):
    """
    Create a linear VTK lookup table from a matplotlib colormap.

    Parameters
    ----------
    name : str
        Name of the matplotlib colormap.
    value_range : tuple
        Range of values to map the colors to.
    n_samples : int
        Number of samples to take in the matplotlib colormap.

    Returns
    -------
    vtkLookupTable
        A VTK lookup table (range: [0, 255]).
    """
    lut = get_lookup_table(name)
    return lut_from_colors(
        lut(np.linspace(0., 1., n_samples)) * 255., value_range)


def clip_and_normalize_data_for_cmap(
        data, clip_outliers=False, min_range=None, max_range=None,
        min_cmap=None, max_cmap=None, log=False, LUT=None):
    """
    Normalizes data between 0 and 1 for an easier management with colormaps.
    The real lower bound and upperbound are returned.

    Data can be clipped to (min_range, max_range) before normalization.
    Alternatively, data can be kept as is, but the colormap be fixed to
    (min_cmap, max_cmap).

    Parameters
    ----------
    data: np.array
        The data: a vector array.
    clip_outliers: bool
        If True, clips the data to the lowest and highest 5% quantile before
        normalizing and before any other clipping.
    min_range: float
        Data values below min_range will be clipped.
    max_range: float
        Data values above max_range will be clipped.
    min_cmap: float
        Minimum value of the colormap. Most useful when min_range and max_range
        are not set; to fix the colormap range without modifying the data.
    max_cmap: float
        Maximum value of the colormap. Idem.
    log: bool
        If True, apply a logarithmic scale to the data.
    LUT: np.ndarray
        If set, replaces the data values by the Look-Up Table values. In order,
        the first value of the LUT is set everywhere where data==1, etc.
    """
    # Make sure data type is float
    if isinstance(data, list):
        data = np.asarray(data)
    data = data.astype(float)

    if LUT is not None:
        for i, val in enumerate(LUT):
            data[data == i+1] = val

    # Clipping
    if clip_outliers:
        data = np.clip(data, np.quantile(data, 0.05),
                       np.quantile(data, 0.95))
    if min_range is not None or max_range is not None:
        data = np.clip(data, min_range, max_range)

    # get data values range
    if min_cmap is not None:
        lbound = min_cmap
    else:
        lbound = np.min(data)
    if max_cmap is not None:
        ubound = max_cmap
    else:
        ubound = np.max(data)

    if log:
        data[data > 0] = np.log10(data[data > 0])

    # normalize data between 0 and 1
    data = (data - lbound) / (ubound - lbound)
    return data, lbound, ubound


def format_hexadecimal_color_to_rgb(color):
    """
    Convert a hexadecimal color name (either "#RRGGBB" or 0xRRGGBB) to RGB
    values.

    Parameters
    ----------
    color: str
        The hexadecimal name

    Returns
    -------
    (R, G, B): int values.
    """
    if len(color) == 7:
        color = '0x' + color.lstrip('#')

    if len(color) == 8:
        color_int = int(color, 0)
        red = color_int >> 16
        green = (color_int & 0x00FF00) >> 8
        blue = color_int & 0x0000FF
    else:
        raise ValueError('Hexadecimal RGB color should be formatted as '
                         '"#RRGGBB" or 0xRRGGBB.')

    return red, green, blue


def prepare_colorbar_figure(cmap, lbound, ubound, nb_values=255, nb_ticks=10,
                            horizontal=False, log=False,):
    """
    Prepares a matplotlib figure of a colorbar.

    Parameters
    ----------
    cmap: plt colormap
        Ex, result from get_colormap().
    lbound: float
        Lower bound
    ubound: float
        Upper bound
    nb_values: int
        Number of values. The cmap will be linearly divided between lbound and
        ubound into nb_values values. Default: 255.
    nb_ticks: int
        The ticks on the colorbar can be set differently than the nb_values.
        Default: 10.
    horizontal: bool
        If true, plot a horizontal cmap.
    log: bool
        If true, apply a logarithm scaling.

    Returns
    -------
    fig: plt figure
        The plt figure.
    """
    gradient = cmap(np.linspace(0, 1, ))[:, 0:3]
    width = int(nb_values * 0.1)
    gradient = np.tile(gradient, (width, 1, 1))
    if not horizontal:
        gradient = np.swapaxes(gradient, 0, 1)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(gradient, origin='lower')

    ticks_labels = ['{0:.3f}'.format(i) for i in
                    np.linspace(lbound, ubound, nb_ticks)]

    if log:
        ticks_labels = ['log(' + t + ')' for t in ticks_labels]

    ticks = np.linspace(0, nb_values - 1, nb_ticks)
    if not horizontal:
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks_labels)
        ax.set_xticks([])
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks_labels)
        ax.set_yticks([])
    return fig
