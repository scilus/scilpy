# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import colors


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
