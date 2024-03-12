# -*- coding: utf-8 -*-
import itertools
import numpy as np


NB_FLIPS = 4
ANGLE_TH = np.pi / 6.

# directions to the 26 neighbors.
# Preparing once rather than in compute_fiber_coherence, possibly called many
# times.
ALL_NEIGHBORS = np.indices((3, 3, 3))
ALL_NEIGHBORS = ALL_NEIGHBORS.T.reshape((27, 3)) - 1
ALL_NEIGHBORS = np.delete(ALL_NEIGHBORS, 13, axis=0)


def compute_coherence_table_for_transforms(directions, values):
    """
    Compute fiber coherence indexes for all possible axes permutations/flips
    (ex, originating from a flip in the gradient table).

    The mathematics are presented in :
    [1] Schilling et al. A fiber coherence index for quality control of B-table
    orientation in diffusion MRI scans. Magn Reson Imaging. 2019 May;58:82-89.
    doi: 10.1016/j.mri.2019.01.018.

    Parameters
    ----------
    directions: ndarray (x, y, z, 3)
        Principal fiber orientation for each voxel.
    values: ndarray (x, y, z)
        Anisotropy measure for each voxel (e.g. FA map).

    Returns
    -------
    coherence: list
        Fiber coherence value for each permutation/flip.
    transforms: list
        Transform representing each permutation/flip, in the same
        order as `coherence` list.
    """
    # Generate transforms for 24 possible permutation/flips of
    # gradient directions. (Reminder. We want to verify if there was possibly
    # an error in the gradient table).
    permutations = list(itertools.permutations([0, 1, 2]))
    transforms = np.zeros((len(permutations)*NB_FLIPS, 3, 3))
    for i in range(len(permutations)):
        transforms[i*NB_FLIPS, np.arange(3), permutations[i]] = 1
        for ii in range(3):
            flip = np.eye(3)
            flip[ii, ii] = -1
            transforms[ii+i*NB_FLIPS+1] = transforms[i*NB_FLIPS].dot(flip)

    # Compute the coherence for each one.
    coherence = []
    for t in transforms:
        index = compute_fiber_coherence(directions.dot(t), values)
        coherence.append(index)
    return coherence, list(transforms)


def compute_fiber_coherence(peaks, values):
    """
    Compute the fiber coherence for `peaks` and `values`.

    Parameters
    ----------
    peaks: ndarray (x, y, z, 3)
        Principal fiber orientation for each voxel.
    values: ndarray (x, y, z)
        Anisotropy measure for each voxel (e.g. FA map).

    Returns
    -------
    coherence: float
        Fiber coherence value.
    """
    # Normalizing peaks
    norm_peaks = np.zeros_like(peaks)
    norms = np.linalg.norm(peaks, axis=-1)
    norm_peaks[norms > 0] = peaks[norms > 0] / norms[norms > 0][..., None]

    coherence = 0.0
    for di in ALL_NEIGHBORS:
        tx, ty, tz = di.astype(int)
        slice_x = slice(1 + tx, peaks.shape[0] - 1 + tx)
        slice_y = slice(1 + ty, peaks.shape[1] - 1 + ty)
        slice_z = slice(1 + tz, peaks.shape[2] - 1 + tz)

        di_norm = di / np.linalg.norm(di)

        # Spatial coherence between the peak at each voxel and the direction to
        # the neighbor di.
        # Ex: if the peak is aligned in x and current di is aligned in x,
        # returns True (with angle < 30 ; cos angle > 30)
        cos_angles = np.abs(norm_peaks.dot(di_norm))
        I_u = cos_angles > np.cos(ANGLE_TH)

        # Doing the same thing with v; results in the same image but translated
        # from one voxel. (With 1 voxel padding around the border).
        I_v = np.zeros_like(I_u)
        I_v[1:-1, 1:-1, 1:-1] = I_u[slice_x, slice_y, slice_z]

        # Where both conditions are met:
        I_uv = np.logical_and(I_u, I_v)
        u = np.nonzero(I_uv)

        # v = the same voxels as u, but with the neighborhood difference.
        v = tuple(np.array(u) + di.astype(int).reshape(3, 1))

        # Summing the FA of those voxels
        coherence += np.sum(values[u]) + np.sum(values[v])

    return coherence
