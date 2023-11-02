# -*- coding: utf-8 -*-

import logging

from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import (mask_for_response_ssst,
                                   response_from_mask_ssst)
from dipy.reconst.mcsd import mask_for_response_msmt, response_from_mask_msmt
from dipy.segment.mask import applymask
import numpy as np

from scilpy.gradients.bvec_bval_tools import (check_b0_threshold,
                                              is_normalized_bvecs, normalize_bvecs)


def compute_ssst_frf(data, bvals, bvecs, mask=None, mask_wm=None,
                     fa_thresh=0.7, min_fa_thresh=0.5, min_nvox=300,
                     roi_radii=10, roi_center=None, force_b0_threshold=False):
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
    roi_radii : int or array-like (3,), optional
        Use those radii to select a cuboid roi to estimate the FRF. The roi
        will be a cuboid spanning from the middle of the volume in each
        direction with the different radii. Defaults to 10.
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

    b0_thr = check_b0_threshold(force_b0_threshold, bvals.min(), bvals.min())

    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_thr)

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

    # Iteratively trying to fit at least min_nvox voxels. Lower the FA
    # threshold when it doesn't work. Fail if the fa threshold is smaller than
    # the min_threshold.
    # We use an epsilon since the -= 0.05 might incur numerical imprecision.
    nvox = 0
    while nvox < min_nvox and fa_thresh >= min_fa_thresh - 0.00001:
        mask = mask_for_response_ssst(gtab, data,
                                      roi_center=roi_center,
                                      roi_radii=roi_radii,
                                      fa_thr=fa_thresh)
        nvox = np.sum(mask)
        response, ratio = response_from_mask_ssst(gtab, data, mask)

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


def compute_msmt_frf(data, bvals, bvecs, btens=None, data_dti=None,
                     bvals_dti=None, bvecs_dti=None, btens_dti=None,
                     mask=None, mask_wm=None, mask_gm=None, mask_csf=None,
                     fa_thr_wm=0.7, fa_thr_gm=0.2, fa_thr_csf=0.1,
                     md_thr_gm=0.0007, md_thr_csf=0.003, min_nvox=300,
                     roi_radii=10, roi_center=None, tol=20,
                     force_b0_threshold=False):
    """Compute a multi-shell, multi-tissue single Fiber
    Response Function from a DWI volume.
    A DTI fit is made, and voxels containing a single fiber population are
    found using a threshold on the FA and MD.

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
        computations and reconstruction.
    mask_wm : ndarray, optional
        3D mask with shape (X,Y,Z)
        Binary white matter mask. Only the data inside this mask will be used
        to estimate the fiber response function of WM.
    mask_gm : ndarray, optional
        3D mask with shape (X,Y,Z)
        Binary grey matter mask. Only the data inside this mask will be used
        to estimate the fiber response function of GM.
    mask_csf : ndarray, optional
        3D mask with shape (X,Y,Z)
        Binary csf mask. Only the data inside this mask will be used to
        estimate the fiber response function of CSF.
    fa_thr_wm : float, optional
        Use this threshold to select single WM fiber voxels from the FA inside
        the WM mask defined by mask_wm. Each voxel above this threshold will be
        selected. Defaults to 0.7
    fa_thr_gm : float, optional
        Use this threshold to select GM voxels from the FA inside the GM mask
        defined by mask_gm. Each voxel below this threshold will be selected.
        Defaults to 0.2
    fa_thr_csf : float, optional
        Use this threshold to select CSF voxels from the FA inside the CSF mask
        defined by mask_csf. Each voxel below this threshold will be selected.
        Defaults to 0.1
    md_thr_gm : float, optional
        Use this threshold to select GM voxels from the MD inside the GM mask
        defined by mask_gm. Each voxel below this threshold will be selected.
        Defaults to 0.0007
    md_thr_csf : float, optional
        Use this threshold to select CSF voxels from the MD inside the CSF mask
        defined by mask_csf. Each voxel below this threshold will be selected.
        Defaults to 0.003
    min_nvox : int, optional
        Minimal number of voxels needing to be identified as single fiber
        voxels in the automatic estimation. Defaults to 300.
    roi_radii : int or array-like (3,), optional
        Use those radii to select a cuboid roi to estimate the FRF. The roi
        will be a cuboid spanning from the middle of the volume in each
        direction with the different radii. Defaults to 10.
    roi_center : tuple(3), optional
        Use this center to span the roi of size roi_radius (center of the
        3D volume).
    tol : int
        tolerance gap for b-values clustering. Defaults to 20
    force_b0_threshold : bool, optional
        If set, will continue even if the minimum bvalue is suspiciously high.

    Returns
    -------
    reponses : list of ndarray
        Fiber Response Function of each (3) tissue type, with shape (4, N).
    frf_masks : list of ndarray
        Mask where the frf was calculated, for each (3) tissue type, with
        shape (X, Y, Z).

    Raises
    ------
    ValueError
        If less than `min_nvox` voxels were found with sufficient FA to
        estimate the FRF.
    """
    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)

    b0_thr = check_b0_threshold(force_b0_threshold, bvals.min(), bvals.min())

    gtab = gradient_table(bvals, bvecs, btens=btens, b0_threshold=b0_thr)

    if data_dti is None and bvals_dti is None and bvecs_dti is None:
        logging.warning(
            "No data specific to DTI was given. If b-values go over 1200, "
            "this might produce wrong results.")
        wm_frf_mask, gm_frf_mask, csf_frf_mask \
            = mask_for_response_msmt(gtab, data,
                                     roi_center=roi_center,
                                     roi_radii=roi_radii,
                                     wm_fa_thr=fa_thr_wm,
                                     gm_fa_thr=fa_thr_gm,
                                     csf_fa_thr=fa_thr_csf,
                                     gm_md_thr=md_thr_gm,
                                     csf_md_thr=md_thr_csf)
    elif (data_dti is not None and bvals_dti is not None
          and bvecs_dti is not None):
        if not is_normalized_bvecs(bvecs_dti):
            logging.warning('Your b-vectors do not seem normalized...')
            bvecs_dti = normalize_bvecs(bvecs_dti)

        check_b0_threshold(force_b0_threshold, bvals_dti.min())
        gtab_dti = gradient_table(bvals_dti, bvecs_dti, btens=btens_dti)

        wm_frf_mask, gm_frf_mask, csf_frf_mask \
            = mask_for_response_msmt(gtab_dti, data_dti,
                                     roi_center=roi_center,
                                     roi_radii=roi_radii,
                                     wm_fa_thr=fa_thr_wm,
                                     gm_fa_thr=fa_thr_gm,
                                     csf_fa_thr=fa_thr_csf,
                                     gm_md_thr=md_thr_gm,
                                     csf_md_thr=md_thr_csf)
    else:
        msg = """Input not valid. Either give no _dti input, or give all
        data_dti, bvals_dti and bvecs_dti."""
        raise ValueError(msg)

    if mask is not None:
        wm_frf_mask *= mask
        gm_frf_mask *= mask
        csf_frf_mask *= mask
    if mask_wm is not None:
        wm_frf_mask *= mask_wm
    if mask_gm is not None:
        gm_frf_mask *= mask_gm
    if mask_csf is not None:
        csf_frf_mask *= mask_csf

    msg = """Could not find at least {0} voxels for the {1} mask. Look at
    previous warnings or be sure that external tissue masks overlap with the
    cuboid ROI."""

    if np.sum(wm_frf_mask) < min_nvox:
        raise ValueError(msg.format(min_nvox, "WM"))
    if np.sum(gm_frf_mask) < min_nvox:
        raise ValueError(msg.format(min_nvox, "GM"))
    if np.sum(csf_frf_mask) < min_nvox:
        raise ValueError(msg.format(min_nvox, "CSF"))

    frf_masks = [wm_frf_mask, gm_frf_mask, csf_frf_mask]

    response_wm, response_gm, response_csf \
        = response_from_mask_msmt(gtab, data, wm_frf_mask, gm_frf_mask,
                                  csf_frf_mask, tol=tol)

    responses = [response_wm, response_gm, response_csf]

    return responses, frf_masks
