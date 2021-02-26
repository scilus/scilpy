# -*- coding: utf-8 -*-
import itertools
import numpy as np


NB_FLIPS = 4
ANGLE_TH = np.pi / 6.


def compute_fiber_coherence_table(directions, values, mask=None):
    """
    Compute fiber coherence indexes for all possible axis permutations/flips.
    """
    permutations = list(itertools.permutations([0, 1, 2]))
    transforms = np.zeros((len(permutations)*NB_FLIPS, 3, 3))

    # Generate transforms for 24 possible permutation/flips of
    # gradient directions
    for i in range(len(permutations)):
        transforms[i*NB_FLIPS, np.arange(3), permutations[i]] = 1
        for ii in range(3):
            flip = np.eye(3)
            flip[ii, ii] = -1
            transforms[ii+i*NB_FLIPS+1] = transforms[i*NB_FLIPS].dot(flip)

    table = {}
    key_id = 0
    for t in transforms:
        c_map = compute_fiber_coherence_map(directions.dot(t), values, mask)
        table['t{0}'.format(key_id)] = {'map': c_map, 't': transforms}
        key_id += 1
    return table


def compute_fiber_coherence_map(peaks, values, mask=None):
    """
    One peak direction per voxel associated to one anisotropy value.
    """
    if mask is not None:
        peaks = peaks * mask.astype(float)[..., None]

    coherence_map = np.zeros(peaks.shape[:3])
    pad_peaks = np.pad(peaks, ((1, 1), (1, 1), (1, 1), (0, 0)))
    pad_vals = np.pad(values, ((1, 1), (1, 1), (1, 1)))
    nonzero_voxels = np.array(np.nonzero(np.abs(peaks).sum(axis=-1))).T
    for vox_coord in nonzero_voxels:
        x, y, z = vox_coord
        win_peaks = pad_peaks[x:x+3, y:y+3, z:z+3]
        win_vals = pad_vals[x:x+3, y:y+3, z:z+3]
        coherence_map[x, y, z] =\
            compute_fiber_coherence(win_peaks, win_vals)

    return coherence_map


def compute_fiber_coherence(win_peaks, win_vals):
    """
    Naive fiber coherence index implementation for one peak direction
    per voxel inside a 3 x 3 x 3 window.
    """
    c = 0
    ax, ay, az = (1, 1, 1)
    vox_a = np.array([ax, ay, az])
    u = win_peaks[ax, ay, az]
    u /= np.linalg.norm(u)
    fu = win_vals[ax, ay, az]

    if np.abs(u).sum() == 0.:
        return 0

    nonzero_vox = np.array(np.nonzero(np.abs(win_peaks).sum(axis=-1))).T
    for vox_b in nonzero_vox:
        bx, by, bz = vox_b
        if not np.array_equal(vox_b, vox_a):
            d = vox_b - vox_a
            d = d / np.linalg.norm(d)

            v = win_peaks[bx, by, bz]
            v /= np.linalg.norm(v)
            fv = win_vals[bx, by, bz]
            if (np.abs(u.dot(d)) > np.cos(ANGLE_TH) and
                    np.abs(v.dot(d)) > np.cos(ANGLE_TH)):
                c += fu + fv
    return c
