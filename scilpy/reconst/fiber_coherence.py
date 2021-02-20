# -*- coding: utf-8 -*-
import itertools
import numpy as np
from scipy.ndimage import correlate
import nibabel as nib


NB_FLIPS = 4
ANGLE_TH = np.pi / 6.


def _generate_key_for_transform(transform):
    key = np.ones(6)
    key[:3] = np.nonzero(transform)[1]
    flip_mask = transform < 0
    if flip_mask.any():
        key[np.nonzero(flip_mask)[1] + 3] *= -1
    return tuple(key.astype(int))


def compute_fiber_coherence_table(directions, values, mask=None):
    """
    Compute fiber coherence indexes for all possible axis permutations/flips.
    """
    permutations = list(itertools.permutations([0, 1, 2]))
    transforms = np.zeros((len(permutations)*NB_FLIPS, 3, 3))

    # Generate 24 possible permutation/flips of gradient directions
    for i in range(len(permutations)):
        transforms[i*NB_FLIPS, np.arange(3), permutations[i]] = 1
        for ii in range(3):
            flip = np.eye(3)
            flip[ii, ii] = -1
            transforms[ii+i*NB_FLIPS+1] = transforms[i*NB_FLIPS].dot(flip)

    table = {}
    for t in transforms:
        key = _generate_key_for_transform(t)
        table[key], _ =\
            compute_fiber_coherence_index(directions.dot(t), values, mask)
    return table


def compute_fiber_coherence_index(directions, values, mask=None):
    """
    C = sum_{u, v} I(|dot(d, u)|>cos(30))*I(|dot(d, v)|>cos(30))*(f(u)+f(v))
    """
    # each voxel has 26 neighbors to check
    # lets first consider the case with only one direction per voxel
    # there is probably a convolution approach for this equation
    # each direction is connected to 2 voxels
    D = _get_directions_to_neighbours()
    norms = np.linalg.norm(directions, axis=-1)
    UV = np.zeros_like(directions)
    UV[norms > 0] = directions[norms > 0] / norms[norms > 0][..., None]

    coherence_map = np.zeros(directions.shape[:3])
    for d in D:
        W = np.zeros((3, 3, 3))
        W[1, 1, 1] = 1
        W[d] = 1
        d_norm = d / np.linalg.norm(d)
        aligned_vox = UV.dot(d_norm) > np.cos(ANGLE_TH)
        aligned_vox =\
            correlate(aligned_vox.astype(float), W, mode='constant') > 1
        coherence_map[aligned_vox] +=\
            correlate(values, W, mode='constant')[aligned_vox]

    return coherence_map.sum(), coherence_map


def _get_directions_to_neighbours():
    d = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if not(i == 1 and j == 1 and k == 1):
                    d.append([i, j, k])
    d = np.array(d, dtype=int) - 1
    return d
