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

    indices = []
    for t in transforms:
        index = compute_fiber_coherence_fast(directions.dot(t), values, mask)
        indices.append(index)
    return indices, list(transforms)


def compute_fiber_coherence_fast(peaks, values, mask=None):
    """
    One peak direction per voxel associated to one anisotropy value.
    """
    if mask is not None:
        peaks = peaks * mask.astype(float)[..., None]

    # directions to neighbors
    all_d = np.indices((3, 3, 3))
    all_d = all_d.T.reshape((27, 3)) - 1
    all_d = np.delete(all_d, 13, axis=0)

    norm_peaks = np.zeros_like(peaks)
    norms = np.linalg.norm(peaks, axis=-1)
    norm_peaks[norms > 0] = peaks[norms > 0] / norms[norms > 0][..., None]

    coherence = 0.0
    for di in all_d:
        tx, ty, tz = di.astype(int)
        slice_x = slice(1 + tx, peaks.shape[0] - 1 + tx)
        slice_y = slice(1 + ty, peaks.shape[1] - 1 + ty)
        slice_z = slice(1 + tz, peaks.shape[2] - 1 + tz)

        di_norm = di / np.linalg.norm(di)
        I_u = np.abs(norm_peaks.dot(di_norm)) > np.cos(ANGLE_TH)
        I_v = np.zeros_like(I_u)
        I_v[1:-1, 1:-1, 1:-1] = I_u[slice_x, slice_y, slice_z]

        I_uv = np.logical_and(I_u, I_v)
        u = np.nonzero(I_uv)
        v = tuple(np.array(u) + di.astype(int).reshape(3, 1))
        coherence += np.sum(values[u]) + np.sum(values[v])

    return coherence
