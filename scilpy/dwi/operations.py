import logging
import math
import pprint

import numpy as np

from scilpy.gradients.bvec_bval_tools import identify_shells, \
    round_bvals_to_shell, DEFAULT_B0_THRESHOLD, is_normalized_bvecs, \
    normalize_bvecs


def apply_bias_field(dwi_data, bias_field_data, mask_data):
    """
    To apply a bias field (computed beforehands), we need to
    1) Divide the dwi by the bias field. This is the correction itself.
    See the following references:
    https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
    https://mrtrix.readthedocs.io/en/dev/reference/commands/dwibiascorrect.html
    2) Rescale the dwi, to ensure that the initial min-max range is kept.

    Parameters
    ----------
    dwi_data: np.ndarray
        The 4D dwi data.
    bias_field_data: np.ndarray
        The 3D bias field. Typically comes from ANTS'S N4BiasFieldCorrection.
    mask_data: np.ndarray
        The mask where to apply the bias field.

    Returns
    -------
    dwi_data: np.ndarray
        The modified 4D dwi_data.
    """
    nuc_dwi_data = np.divide(
        dwi_data[mask_data, :], bias_field_data[mask_data][:, None])

    rescaled_nuc_data = _rescale_dwi(dwi_data[mask_data], nuc_dwi_data)
    dwi_data[mask_data] = rescaled_nuc_data

    return dwi_data


def _rescale_intensity(val, slope, in_max, bc_max):
    """
    Rescale an intensity value given a scaling factor.
    This scaling factor ensures that the intensity
    range before and after correction is the same.

    Parameters
    ----------
    val: float
         Value to be scaled
    slope: float
         Scaling factor to be applied
    in_max: float
         Max possible value
    bc_max: float
         Max value in the bias correction value range

    Returns
    -------
    rescaled_value: float
         Bias field corrected value scaled by the slope
         of the data
    """

    return in_max - slope * (bc_max - val)


# https://github.com/stnava/ANTs/blob/master/Examples/N4BiasFieldCorrection.cxx
def _rescale_dwi(in_data, bc_data):
    """
    Apply N4 Bias Field Correction to a DWI volume.
    bc stands for bias correction. The code comes
    from the C++ ANTS implmentation.

    Parameters
    ----------
    in_data: ndarray (x, y, z, ndwi)
         Input DWI volume 4-dimensional data.
    bc_data: ndarray (x, y, z, ndwi)
         Bias field correction volume estimated from ANTS
         Copied for every dimension of the DWI 4-th dimension

    Returns
    -------
    bc_data: ndarray (x, y, z, ndwi)
         Bias field corrected DWI volume
    """

    in_min = np.amin(in_data)
    in_max = np.amax(in_data)
    bc_min = np.amin(bc_data)
    bc_max = np.amax(bc_data)

    slope = (in_max - in_min) / (bc_max - bc_min)

    chunk = np.arange(0, len(in_data), 100000)
    chunk = np.append(chunk, len(in_data))
    for i in range(len(chunk) - 1):
        nz_bc_data = bc_data[chunk[i]:chunk[i + 1]]
        rescale_func = np.vectorize(_rescale_intensity, otypes=[np.float32])

        rescaled_data = rescale_func(nz_bc_data, slope, in_max, bc_max)
        bc_data[chunk[i]:chunk[i + 1]] = rescaled_data

    return bc_data


def compute_dwi_attenuation(dwi_weights: np.ndarray, b0: np.ndarray):
    """ Compute signal attenuation by dividing the dwi signal with the b0.

    Parameters:
    -----------
    dwi_weights : np.ndarray of shape (X, Y, Z, #gradients)
        Diffusion weighted images.
    b0 : np.ndarray of shape (X, Y, Z)
        B0 image.

    Returns
    -------
    dwi_attenuation : np.ndarray
        Signal attenuation (Diffusion weights normalized by the B0).
    """
    # Avoid division by 0. Remember coordinates where the b0 was 0. We will set
    # those voxels to 0 in the final result.
    zeros_mask = b0 == 0

    b0 = b0[..., None]  # Easier to work if it is a 4D array.

    # Make sure that, in every voxel, weights are lower in the dwi than in the
    # b0. Should always be the case, but with the noise we never know!
    erroneous_voxels = np.any(dwi_weights > b0, axis=3)
    nb_erroneous_voxels = np.sum(erroneous_voxels)
    if nb_erroneous_voxels != 0:
        logging.info("# of voxels where `dwi_signal > b0` in any direction: "
                     "{}. They were set to the b0 value to allow computing "
                     "signal attenuation."
                     .format(nb_erroneous_voxels))
        dwi_weights = np.minimum(dwi_weights, b0)

    # Compute attenuation
    b0[zeros_mask] = 1e-10
    dwi_attenuation = dwi_weights / b0
    dwi_attenuation *= ~zeros_mask[:, :, :, None]

    return dwi_attenuation


def detect_volume_outliers(data, bvals, bvecs, std_scale,
                           b0_thr=DEFAULT_B0_THRESHOLD):
    """
    Detects outliers. Finds the 3 closest angular neighbors of each direction
    (per shell) and computes the voxel-wise correlation.
    If the angles or correlations to neighbors are below the shell average (by
    std_scale x STD) it will flag the volume as a potential outlier.

    Parameters
    ----------
    data: np.ndarray
        4D Input diffusion volume with shape (X, Y, Z, N)
    bvals : ndarray
        1D bvals array with shape (N,)
    bvecs : ndarray
        2D bvecs array with shape (N, 3)
    std_scale: float
        How many deviation from the mean are required to be considered an
        outlier.
    b0_thr: float
        Value below which b-values are considered as b0.

    Returns
    -------
    results_dict: dict
        The resulting statistics.
        One key per shell (its b-value). For each key, the associated entry is
        an array of shape [nb_points, 3] where columns are:
            - point_idx: int, the index of the bvector in the input bvecs.
            - mean_angle: float, the mean angles of the 3 closest bvecs, in
              degree
            - mean_correlation: float, the mean correlation of the 3D data
            associated to the 3 closest bvecs.
    outliers_dict: dict
        The resulting outliers.
        One key per shell (its b-value). For each key, the associated entry is
        a dict {'outliers_angle': list[int],
                'outliers_corr': list[int]}
        The indices of outliers (indices in the original bvecs).
    """
    if not is_normalized_bvecs(bvecs):
        logging.warning("Your b-vectors do not seem normalized... Normalizing")
        bvecs = normalize_bvecs(bvecs)

    results_dict = {}
    shells_to_extract = identify_shells(bvals, b0_thr, sort=True)[0]
    bvals = round_bvals_to_shell(bvals, shells_to_extract)
    for bval in shells_to_extract[shells_to_extract > b0_thr]:
        shell_idx = np.where(bvals == bval)[0]

        # Requires at least 3 values per shell to find 3 closest values!
        # Requires at least 5 values to use argpartition, below.
        if len(shell_idx) < 5:
            raise NotImplementedError(
                "This outlier detection method is only available with at "
                "least 5 points per shell. Got {} on shell {}."
                .format(len(shell_idx), bval))

        shell = bvecs[shell_idx, :]  # All bvecs on that shell
        results_dict[bval] = np.ones((len(shell), 3)) * -1
        for i, vec in enumerate(shell):
            # Supposing that vectors are normalized, cos(angle) = dot
            dot_product = np.clip(np.tensordot(shell, vec, axes=1), -1, 1)
            angles = np.rad2deg(np.arccos(dot_product))
            angles[np.isnan(angles)] = 0

            # Managing the symmetry between b-vectors:
            # if angle is > 90, it becomes 180 - x
            big_angles = angles > 90
            angles[big_angles] = 180 - angles[big_angles]

            # Using argpartition rather than sort; faster. With kth=4, the 4th
            # element is correctly positioned, and smaller elements are
            # placed before. Considering that we will then remove the b-vec
            # itself (angle 0), we are left with the 3 closest angles in
            # idx[0:3] (not necessarily sorted, but ok).
            idx = np.argpartition(angles, 4).tolist()
            idx.remove(i)

            avg_angle = np.average(angles[idx[:3]])

            corr = np.corrcoef([data[..., shell_idx[i]].ravel(),
                                data[..., shell_idx[idx[0]]].ravel(),
                                data[..., shell_idx[idx[1]]].ravel(),
                                data[..., shell_idx[idx[2]]].ravel()])
            # Corr is a triangular matrix. The interesting line is the first:
            # current data vs the 3 others. First value is with itself = 1.
            results_dict[bval][i] = [shell_idx[i], avg_angle,
                                     np.average(corr[0, 1:])]

    # Computation done. Now verifying if above scale.
    # Loop on shells:
    logging.info("Analysing, for each bvec, the mean angle of the 3 closest "
                 "bvecs, and the mean correlation of their associated data.")
    outliers_dict = {}
    for key in results_dict.keys():
        # Column #1 = The mean_angle for all bvecs
        avg_angle = np.average(results_dict[key][:, 1])
        std_angle = np.std(results_dict[key][:, 1])

        # Column #2 = The mean_corr for all bvecs
        avg_corr = np.average(results_dict[key][:, 2])
        std_corr = np.std(results_dict[key][:, 2])

        # Only looking if some data are *below* the average - n*std.
        outliers_angle = np.argwhere(
            results_dict[key][:, 1] < avg_angle - (std_scale * std_angle))
        outliers_corr = np.argwhere(
            results_dict[key][:, 2] < avg_corr - (std_scale * std_corr))

        logging.info('Results for shell {} with {} directions:'
                     .format(key, len(results_dict[key])))
        logging.info('AVG and STD of angles: {:.2f} +/- {:.2f}'
                     .format(avg_angle, std_angle))
        logging.info('AVG and STD of correlations: {:.4f} +/- {:.4f}'
                     .format(avg_corr, std_corr))

        if len(outliers_angle) or len(outliers_corr):
            logging.info('Possible outliers ({} STD below average):'
                         .format(std_scale))
            if len(outliers_angle):
                logging.info('Outliers based on angle [position (4D), value]')
                for i in outliers_angle:
                    logging.info("   {}".format(results_dict[key][i, 0::2]))
            if len(outliers_corr):
                logging.info('Outliers based on correlation [position (4D), '
                             'value]')
                for i in outliers_corr:
                    logging.info("   {}".format(results_dict[key][i, 0::2]))
        else:
            logging.info('No outliers detected.')

        outliers_dict[key] = {
            'outliers_angle': results_dict[key][outliers_angle, 0],
            'outliers_corr': results_dict[key][outliers_corr, 0]}
        logging.debug('Shell with b-value {}'.format(key))
        logging.debug("\n" + pprint.pformat(results_dict[key]))
        print()

    return results_dict, outliers_dict


def compute_residuals(predicted_data, real_data, b0s_mask=None, mask=None):
    """
    Computes the residuals, a 3D map allowing comparison between the predicted
    data and the real data. The result is the average of differences for each
    feature of the data, on the last axis.

    If data is a tensor, the residuals computation was introduced in:
    [J-D Tournier, S. Mori, A. Leemans. Diffusion Tensor Imaging and Beyond.
    MRM 2011].

    Parameters
    ----------
    predicted_data: np.ndarray
        4D dwi volume.
    real_data: np.ndarray
        4D dwi volume.
    b0s_mask: np.array, optional
        Vector of booleans containing True at indices where the dwi data was a
        b0 (on last dimension). If given, the b0s are rejected from the data
        before computing the residuals.
    mask: np.ndaray, optional
        3D volume. If given, residual is set to 0 outside the mask.
    """
    diff_data = np.abs(predicted_data - real_data)

    if b0s_mask is not None:
        res = np.mean(diff_data[..., ~b0s_mask], axis=-1)
    else:
        res = np.mean(diff_data, axis=-1)

    # See in dipy.reconst.dti: They offer various weighting techniques, not
    # offered here. We also previously offered to normalize the results (to
    # avoid large values), but this option has been removed.

    if mask is not None:
        res *= mask

    return res, diff_data


def compute_residuals_statistics(diff_data):
    """
    Compute statistics on the residuals for each gradient.

    Parameters
    ----------
    diff_data: np.ndarray
        The 4D residuals between the DWI and predicted data.
    """
    # mean residual per DWI
    R_k = np.zeros(diff_data.shape[-1], dtype=np.float32)
    # std residual per DWI
    std = np.zeros(diff_data.shape[-1], dtype=np.float32)
    # first quartile per DWI
    q1 = np.zeros(diff_data.shape[-1], dtype=np.float32)
    # third quartile per DWI
    q3 = np.zeros(diff_data.shape[-1], dtype=np.float32)
    # interquartile per DWI
    iqr = np.zeros(diff_data.shape[-1], dtype=np.float32)

    for k in range(diff_data.shape[-1]):
        x = diff_data[..., k]
        R_k[k] = np.mean(x)
        std[k] = np.std(x)
        q3[k], q1[k] = np.percentile(x, [75, 25])
        iqr[k] = q3[k] - q1[k]

    return R_k, q1, q3, iqr, std
