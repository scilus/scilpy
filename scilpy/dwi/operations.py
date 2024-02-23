import logging
import math
import pprint

import numpy as np

from scilpy.gradients.bvec_bval_tools import identify_shells, \
    round_bvals_to_shell, DEFAULT_B0_THRESHOLD


def apply_bias_field(dwi_data, bias_field_data, mask_data):
    """
    ToDo: Explain formula why applying field = dividing?
     + why we need to rescale after?

    Parameters
    ----------
    dwi_data: np.ndarray
        The 4D dwi data.
    bias_field_data: np.ndarray
        The 3D bias field.
    mask_data: np.ndarray
        The mask where to apply the bias field.

    Returns
    -------
    dwi_data: np.ndarray
        The modified 4D dwi_data.
    """
    nuc_dwi_data = np.divide(
        dwi_data[mask_data],
        bias_field_data[mask_data].reshape((len(mask_data[0]), 1)))

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
    scale: float
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
    for i in range(len(chunk)-1):
        nz_bc_data = bc_data[chunk[i]:chunk[i+1]]
        rescale_func = np.vectorize(_rescale_intensity, otypes=[np.float32])

        rescaled_data = rescale_func(nz_bc_data, slope, in_max, bc_max)
        bc_data[chunk[i]:chunk[i+1]] = rescaled_data

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
    b0 = b0[..., None]  # Easier to work if it is a 4D array.

    # Make sure that, in every voxels, weights are lower in the b0. Should
    # always be the case, but with the noise we never know!
    erroneous_voxels = np.any(dwi_weights > b0, axis=3)
    nb_erroneous_voxels = np.sum(erroneous_voxels)
    if nb_erroneous_voxels != 0:
        logging.info("# of voxels where `dwi_signal > b0` in any direction: "
                     "{}".format(nb_erroneous_voxels))
        dwi_weights = np.minimum(dwi_weights, b0)

    # Compute attenuation
    dwi_attenuation = dwi_weights / b0

    # Make sure we didn't divide by 0.
    dwi_attenuation[np.logical_not(np.isfinite(dwi_attenuation))] = 0.

    return dwi_attenuation


def detect_volume_outliers(data, bvecs, bvals, std_scale,
                           b0_thr=DEFAULT_B0_THRESHOLD):
    """
    Parameters
    ----------
    data: np.ndarray
        The dwi data.
    bvecs: np.ndarray
        The bvecs
    bvals: np.array
        The b-values vector.
    std_scale: float
        How many deviation from the mean are required to be considered an
        outlier.
    b0_thr: float
        Value below which b-values are considered as b0.
    """
    results_dict = {}
    shells_to_extract = identify_shells(bvals, b0_thr, sort=True)[0]
    bvals = round_bvals_to_shell(bvals, shells_to_extract)
    for bval in shells_to_extract[shells_to_extract > b0_thr]:
        shell_idx = np.where(bvals == bval)[0]
        shell = bvecs[shell_idx]
        results_dict[bval] = np.ones((len(shell), 3)) * -1
        for i, vec in enumerate(shell):
            if np.linalg.norm(vec) < 0.001:
                continue

            dot_product = np.clip(np.tensordot(shell, vec, axes=1), -1, 1)
            angle = np.arccos(dot_product) * 180 / math.pi
            angle[np.isnan(angle)] = 0
            idx = np.argpartition(angle, 4).tolist()
            idx.remove(i)

            avg_angle = np.average(angle[idx[:3]])
            corr = np.corrcoef([data[..., shell_idx[i]].ravel(),
                                data[..., shell_idx[idx[0]]].ravel(),
                                data[..., shell_idx[idx[1]]].ravel(),
                                data[..., shell_idx[idx[2]]].ravel()])
            results_dict[bval][i] = [shell_idx[i], avg_angle,
                                     np.average(corr[0, 1:])]

    for key in results_dict.keys():
        avg_angle = np.round(np.average(results_dict[key][:, 1]), 4)
        std_angle = np.round(np.std(results_dict[key][:, 1]), 4)

        avg_corr = np.round(np.average(results_dict[key][:, 2]), 4)
        std_corr = np.round(np.std(results_dict[key][:, 2]), 4)

        outliers_angle = np.argwhere(
            results_dict[key][:, 1] < avg_angle - (std_scale * std_angle))
        outliers_corr = np.argwhere(
            results_dict[key][:, 2] < avg_corr - (std_scale * std_corr))

        logging.info('Results for shell {} with {} directions:'
                     .format(key, len(results_dict[key])))
        logging.info('AVG and STD of angles: {} +/- {}'
                     .format(avg_angle, std_angle))
        logging.info('AVG and STD of correlations: {} +/- {}'
                     .format(avg_corr, std_corr))

        if len(outliers_angle) or len(outliers_corr):
            logging.info('Possible outliers ({} STD below or above average):'
                  .format(std_scale))
            logging.info('Outliers based on angle [position (4D), value]')
            for i in outliers_angle:
                logging.info(results_dict[key][i, :][0][0:2])
            logging.info('Outliers based on correlation [position (4D), ' +
                         'value]')
            for i in outliers_corr:
                logging.info(results_dict[key][i, :][0][0::2])
        else:
            logging.info('No outliers detected.')

        logging.debug('Shell with b-value {}'.format(key))
        logging.debug("\n" + pprint.pformat(results_dict[key]))
        print()


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
