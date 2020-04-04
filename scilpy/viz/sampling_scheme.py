# -*- coding: utf-8 -*-

import logging
import numpy as np
from tempfile import mkstemp

from dipy.data import get_sphere
from fury import actor, window
import fury

vtkcolors = [window.colors.blue,
             window.colors.red,
             window.colors.yellow,
             window.colors.purple,
             window.colors.cyan,
             window.colors.green,
             window.colors.orange,
             window.colors.white,
             window.colors.brown,
             window.colors.grey]


def plot_each_shell(ms, centroids, plot_sym_vecs=True, use_sphere=True, same_color=False,
                    rad=0.025, opacity=1.0, ofile=None, ores=(300, 300)):
    """
    Plot each shell

    Parameters
    ----------
    ms: list of numpy.ndarray
        bvecs for each bval
    plot_sym_vecs: boolean
        Plot symmetrical vectors
    use_sphere: boolean
        rendering of the sphere
    same_color: boolean
        use same color for all shell
    rad: float
        radius of each point
    opacity: float
        opacity for the shells
    ofile: str
        output filename
    ores: tuple
        resolution of the output png

    Return
    ------
    """
    global vtkcolors
    if len(ms) > 10:
        vtkcolors = fury.colormap.distinguishable_colormap(nb_colors=len(ms))

    if use_sphere:
        sphere = get_sphere('symmetric724')
        shape = (1, 1, 1, sphere.vertices.shape[0])
        fid, fname = mkstemp(suffix='_odf_slicer.mmap')
        odfs = np.memmap(fname, dtype=np.float64, mode='w+', shape=shape)
        odfs[:] = 1
        odfs[..., 0] = 1
        affine = np.eye(4)

    for i, shell in enumerate(ms):
        logging.info('Showing shell {}'.format(int(centroids[i])))
        if same_color:
            i = 0
        ren = window.Renderer()
        ren.SetBackground(1, 1, 1)
        if use_sphere:
            sphere_actor = actor.odf_slicer(odfs, affine, sphere=sphere,
                                            colormap='winter', scale=1.0,
                                            opacity=opacity)
            ren.add(sphere_actor)
        pts_actor = actor.point(shell, vtkcolors[i], point_radius=rad)
        ren.add(pts_actor)
        if plot_sym_vecs:
            pts_actor = actor.point(-shell, vtkcolors[i], point_radius=rad)
            ren.add(pts_actor)
        window.show(ren)

        if ofile:
            window.snapshot(ren, fname=ofile + '_shell_' +
                            str(int(centroids[i])) + '.png',
                            size=ores)


def plot_proj_shell(ms, use_sym=True, use_sphere=True, same_color=False,
                    rad=0.025, opacity=1.0, ofile=None, ores=(300, 300)):
    """
    Plot each shell

    Parameters
    ----------
    ms: list of numpy.ndarray
        bvecs for each bvalue
    use_sym: boolean
        Plot symmetrical vectors
    use_sphere: boolean
        rendering of the sphere
    same_color: boolean
        use same color for all shell
    rad: float
        radius of each point
    opacity: float
        opacity for the shells
    ofile: str
        output filename
    ores: tuple
        resolution of the output png

    Return
    ------
    """
    global vtkcolors
    if len(ms) > 10:
        vtkcolors = fury.colormap.distinguishable_colormap(nb_colors=len(ms))

    ren = window.Renderer()
    ren.SetBackground(1, 1, 1)
    if use_sphere:
        sphere = get_sphere('symmetric724')
        shape = (1, 1, 1, sphere.vertices.shape[0])
        fid, fname = mkstemp(suffix='_odf_slicer.mmap')
        odfs = np.memmap(fname, dtype=np.float64, mode='w+', shape=shape)
        odfs[:] = 1
        odfs[..., 0] = 1
        affine = np.eye(4)
        sphere_actor = actor.odf_slicer(odfs, affine, sphere=sphere,
                                        colormap='winter', scale=1.0,
                                        opacity=opacity)

        ren.add(sphere_actor)

    for i, shell in enumerate(ms):
        if same_color:
            i = 0
        pts_actor = actor.point(shell, vtkcolors[i], point_radius=rad)
        ren.add(pts_actor)
        if use_sym:
            pts_actor = actor.point(-shell, vtkcolors[i], point_radius=rad)
            ren.add(pts_actor)
    window.show(ren)
    if ofile:
        window.snapshot(ren, fname=ofile + '.png', size=ores)


def build_ms_from_shell_idx(bvecs, shell_idx):
    """
    Get bvecs from indexes

    Parameters
    ----------
    bvecs: numpy.ndarray
        bvecs
    shell_idx: numpy.ndarray
        index for each bval

    Return
    ------
    ms: list of numpy.ndarray
        bvecs for each bval
    """

    S = len(set(shell_idx))
    if (-1 in set(shell_idx)):
        S -= 1

    ms = []
    for i_ms in range(S):
        ms.append(bvecs[shell_idx == i_ms])

    return ms
