# -*- coding: utf-8 -*-

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


def plot_each_shell(ms, plot_sym_vecs=True, use_sphere=True, same_color=False,
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
            window.snapshot(ren, fname=ofile + '_shell_' + str(i) + '.png',
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


def build_shell_idx_from_bval(bvals, shell_th=50):
    """
    Get index of each bvals

    Parameters
    ----------
    bvals: numpy.ndarray
        array of bvalues
    shell_th: int
        shells threshold

    Return
    ------
    shell_idx: numpy.ndarray
        index for each bval
    """
    target_bvals = _find_target_bvals(bvals, shell_th=shell_th)

    # Pop b0
    if target_bvals[0] < shell_th:
        target_bvals.pop(0)

    shell_idx = _find_shells(bvals, target_bvals, shell_th=shell_th)

    return shell_idx


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


def _find_target_bvals(bvals, shell_th=50):
    """
    Find bvals

    Parameters
    ----------
    bvals: numpy.ndarray
        array of bvals
    shell_th: int
        threshold used to find bvals

    Return
    ------
    target_bvals: list
        unique bvals
    """

    target_bvals = []
    tmp_targets = []
    bvalues = np.unique(bvals)
    distances = np.ediff1d(bvalues) <= shell_th
    wasClose = False

    for idx, distance in enumerate(distances):
        if distance:
            if wasClose:
                tmp_targets[-1].append(bvalues[idx+1])
            else:
                tmp_targets.append([bvalues[idx], bvalues[idx+1]])
            wasClose = True
        else:
            if not(wasClose):
                target_bvals.append(bvalues[idx])
            wasClose = False

    return target_bvals


def _find_shells(bvals, target_bvals, shell_th=50):
    """
    Assign bvecs to a target shell

    Parameters
    ----------
    bvals: numpy.ndarray
        bvals
    target_bvals: list
        list of targeted bvals
    shell_th: int
        Threshold used to select bvals

    Return
    ------
    shells: numpy.ndarray
        Selected shells
    """

    # Not robust
    # shell -1 means nbvecs not part of target_bvals
    shells = -1 * np.ones_like(bvals)

    for shell_id, bval in enumerate(target_bvals):
        shells[(bvals <= bval + shell_th) &
               (bvals >= bval - shell_th)] = shell_id

    return shells
