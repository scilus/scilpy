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
