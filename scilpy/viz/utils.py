# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


RAS_AXES = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2
}


def get_colormap(name):
    """Get a matplotlib colormap from a name or a list of named colors.

    Parameters
    ----------
    name : str
        Name of the colormap or a list of named colors (separated by a -).

    Returns
    -------
    matplotlib.colors.Colormap
        The colormap
    """

    if '-' in name:
        name_list = name.split('-')
        colors_list = [colors.to_rgba(color)[0:3] for color in name_list]
        cmap = colors.LinearSegmentedColormap.from_list('CustomCmap',
                                                        colors_list)
        return cmap

    return plt.colormaps.get_cmap(name)


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

    # TODO: Is there a better way to draw a gradient-filled rectangle?
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


def clip_and_normalize_data_for_cmap(
        data, clip_outliers=False, min_range=None, max_range=None,
        min_cmap=None, max_cmap=None, log=False, LUT=None):
    """
    Normalizes data between 0 and 1 for an easier management with colormaps.
    The real lower bound and upperbound are returned.

    Data can be clipped to (min_range, max_range) before normalization.
    Alternatively, data can be kept as is,

    Parameters
    ----------
    data: np.ndarray
        The data.
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
    data -= lbound
    data = data / ubound if ubound > 0 else data
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
