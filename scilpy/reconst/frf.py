# -*- coding: utf-8 -*-

import logging

from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import auto_response
from dipy.segment.mask import applymask
import numpy as np

from scilpy.utils.bvec_bval_tools import (check_b0_threshold,
                                          is_normalized_bvecs, normalize_bvecs)


def compute_ssst_frf(data, bvals, bvecs, mask=None, mask_wm=None,
                     fa_thresh=0.7, min_fa_thresh=0.5, min_nvox=300,
                     roi_radius=10, roi_center=None, force_b0_threshold=False):
    """Compute a single-shell (under b=1500), single-tissue single Fiber
    Response Function from a DWI volume.
    A DTI fit is made, and voxels containing a single fiber population are
    found using a threshold on the FA.

    Parameters
    ----------
    data : ndarray
        4D Input diffusion volume with shape (X, Y, Z, N)
    bvals : ndarray
        1D bvals array with shape (N,)
    bvecs : ndarray
        2D bvecs array with shape (N, 3)
    mask : ndarray, optional
        3D mask with shape (X,Y,Z)
        Binary mask. Only the data inside the mask will be used for
        computations and reconstruction. Useful if no white matter mask is
        available.
    mask_wm : ndarray, optional
        3D mask with shape (X,Y,Z)
        Binary white matter mask. Only the data inside this mask and above the
        threshold defined by fa_thresh will be used to estimate the fiber
        response function.
    fa_thresh : float, optional
        Use this threshold as the initial threshold to select single fiber
        voxels. Defaults to 0.7
    min_fa_thresh : float, optional
        Minimal value that will be tried when looking for single fiber voxels.
        Defaults to 0.5
    min_nvox : int, optional
        Minimal number of voxels needing to be identified as single fiber
        voxels in the automatic estimation. Defaults to 300.
    roi_radius : int, optional
        Use this radius to select single fibers from the tensor to estimate
        the FRF. The roi will be a cube spanning from the middle of the volume
        in each direction. Defaults to 10.
    roi_center : tuple(3), optional
        Use this center to span the roi of size roi_radius (center of the
        3D volume).
    force_b0_threshold : bool, optional
        If set, will continue even if the minimum bvalue is suspiciously high.

    Returns
    -------
    full_reponse : ndarray
        Fiber Response Function, with shape (4,)

    Raises
    ------
    ValueError
        If less than `min_nvox` voxels were found with sufficient FA to
        estimate the FRF.
    """
    if min_fa_thresh < 0.4:
        logging.warning(
            "Minimal FA threshold ({:.2f}) seems really small. "
            "Make sure it makes sense for this dataset.".format(min_fa_thresh))

    if not is_normalized_bvecs(bvecs):
        logging.warning("Your b-vectors do not seem normalized...")
        bvecs = normalize_bvecs(bvecs)

    check_b0_threshold(force_b0_threshold, bvals.min())

    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    if mask is not None:
        data = applymask(data, mask)

    if mask_wm is not None:
        data = applymask(data, mask_wm)
    else:
        logging.warning(
            "No white matter mask specified! Only mask will be used "
            "(if it has been supplied). \nBe *VERY* careful about the "
            "estimation of the fiber response function to ensure no invalid "
            "voxel was used.")

    # Iteratively trying to fit at least min_nvox voxels. Lower the FA threshold
    # when it doesn't work. Fail if the fa threshold is smaller than
    # the min_threshold.
    # We use an epsilon since the -= 0.05 might incur numerical imprecision.
    nvox = 0
    while nvox < min_nvox and fa_thresh >= min_fa_thresh - 0.00001:
        response, ratio, nvox = auto_response(gtab, data,
                                              roi_center=roi_center,
                                              roi_radius=roi_radius,
                                              fa_thr=fa_thresh,
                                              return_number_of_voxels=True)

        logging.debug(
            "Number of indices is {:d} with threshold of {:.2f}".format(
                nvox, fa_thresh))
        fa_thresh -= 0.05

    if nvox < min_nvox:
        raise ValueError(
            "Could not find at least {:d} voxels with sufficient FA "
            "to estimate the FRF!".format(min_nvox))

    logging.debug(
        "Found {:d} voxels with FA threshold {:.2f} for "
        "FRF estimation".format(nvox, fa_thresh + 0.05))
    logging.debug("FRF eigenvalues: {}".format(str(response[0])))
    logging.debug("Ratio for smallest to largest eigen value "
                  "is {:.3f}".format(ratio))
    logging.debug("Mean of the b=0 signal for voxels used "
                  "for FRF: {}".format(response[1]))

    full_response = np.array([response[0][0], response[0][1],
                              response[0][2], response[1]])

    return full_response
