# -*- coding: utf-8 -*-
import itertools

from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
import numpy as np
from tqdm import tqdm


NB_FLIPS = 4
ANGLE_TH = np.pi / 6.

# directions to the 26 neighbors.
# Preparing once rather than in compute_fiber_coherence, possibly called many
# times.
ALL_NEIGHBORS = np.indices((3, 3, 3))
ALL_NEIGHBORS = ALL_NEIGHBORS.T.reshape((27, 3)) - 1
ALL_NEIGHBORS = np.delete(ALL_NEIGHBORS, 13, axis=0)


def generate_coherence_transforms():
    """
    Generate the 24 possible axis permutation/flip transforms for a
    gradient table (used to check for sign flips or axes swaps).

    Returns
    -------
    transforms: ndarray (24, 3, 3)
        One 3x3 transform per permutation/flip.
    """
    permutations = list(itertools.permutations([0, 1, 2]))
    transforms = np.zeros((len(permutations)*NB_FLIPS, 3, 3))
    for i in range(len(permutations)):
        transforms[i*NB_FLIPS, np.arange(3), permutations[i]] = 1
        for ii in range(3):
            flip = np.eye(3)
            flip[ii, ii] = -1
            transforms[ii+i*NB_FLIPS+1] = transforms[i*NB_FLIPS].dot(flip)
    return transforms


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
    transforms = generate_coherence_transforms()

    # Compute the coherence for each one.
    coherence = []
    for t in transforms:
        index = compute_fiber_coherence(directions.dot(t), values)
        coherence.append(index)
    return coherence, list(transforms)


def find_best_gradient_correction(data, bvals, bvecs, fa, mask,
                                  b0_threshold, verbose=True):
    """
    Refit the DTI model under all 24 axis permutations/flips of bvecs,
    returning the transform that maximizes fiber coherence.

    Contrary to compute_coherence_table_for_transforms, which rotates
    already-fitted peaks, this refits the tensor model from scratch for
    each candidate transform, which is more robust but more expensive.

    Parameters
    ----------
    data: ndarray (X, Y, Z, N)
        DWI data.
    bvals: ndarray
        B-values.
    bvecs: ndarray (N, 3)
        B-vectors to validate.
    fa: ndarray (X, Y, Z)
        FA map, used to weight the coherence computation.
    mask: ndarray (X, Y, Z), optional
        Voxels to fit (e.g. a high-FA mask). If None, all voxels are used.
    b0_threshold: float
        B0 threshold used to rebuild the gradient table for each candidate.
    verbose: bool, optional
        If True, show a progress bar. Default: True.

    Returns
    -------
    best_t: ndarray (3, 3)
        Best-scoring transform (identity if bvecs were already correct).
    best_coherence: float
        Coherence value obtained with best_t.
    """
    transforms = generate_coherence_transforms()
    min_signal = np.min(data[data > 0])

    best_coherence = -1
    best_t = None
    iterator = tqdm(transforms) if verbose else transforms
    for t in iterator:
        bvecs_candidate = np.dot(bvecs, t)
        gtab_candidate = gradient_table(bvals, bvecs=bvecs_candidate,
                                        b0_threshold=b0_threshold)
        tenmodel_candidate = TensorModel(gtab_candidate, fit_method='WLS',
                                         min_signal=min_signal)
        tenfit_candidate = tenmodel_candidate.fit(data, mask=mask)

        # evecs is (X, Y, Z, 3, 3), evecs[..., 0] is the first eigenvector
        # (principal direction).
        peaks = tenfit_candidate.evecs[..., 0]
        coherence = compute_fiber_coherence(peaks, fa)

        if coherence > best_coherence:
            best_coherence = coherence
            best_t = t

    return best_t, best_coherence


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
