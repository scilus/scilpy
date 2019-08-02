from __future__ import division

import numpy as np
from tempfile import mkstemp

from dipy.data import get_sphere
from fury import actor, window

# TODO: Make it robust to more than 10 b-values
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


def plot_each_shell(ms, use_sym=True, use_sphere=True, same_color=False,
                    rad=0.025, opacity=1.0):
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
        if use_sym:
            pts_actor = actor.point(-shell, vtkcolors[i], point_radius=rad)
            ren.add(pts_actor)
        window.show(ren)


def plot_proj_shell(ms, use_sym=True, use_sphere=True, same_color=False,
                    rad=0.025, opacity=1.0):

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


def build_shell_idx_from_bval(bvals, shell_th=50):
    target_bvalues = _find_target_bvalues(bvals, shell_th=shell_th)

    # Pop b0
    if target_bvalues[0] < shell_th:
        target_bvalues.pop(0)

    shell_idx = _find_shells(bvals, target_bvalues, shell_th=shell_th)

    return shell_idx


def build_ms_from_shell_idx(bvecs, shell_idx):
    S = len(set(shell_idx))
    if (-1 in set(shell_idx)):
        S -= 1

    ms = []
    for i_ms in range(S):
        ms.append(bvecs[shell_idx == i_ms])

    return ms


# Attempt to find the b-values of the shells
def _find_target_bvalues(bvals, shell_th=50):
    # Not robust
    target_bvalues = []

    bvalues = np.sort(np.array(list(set(bvals))))

    for bval in bvalues:
        add_bval = True
        for target_bval in target_bvalues:
            if (bval <= target_bval + shell_th) & \
               (bval >= target_bval - shell_th):
                add_bval = False
        if add_bval:
            target_bvalues.append(bval)

    return target_bvalues


# Assign bvecs to a target shell
def _find_shells(bvals, target_bvalues, shell_th=50):
    # Not robust
    # shell -1 means nbvecs not part of target_bvalues
    shells = -1 * np.ones_like(bvals)

    for shell_id, bval in enumerate(target_bvalues):
        shells[(bvals <= bval + shell_th) &
               (bvals >= bval - shell_th)] = shell_id

    return shells
