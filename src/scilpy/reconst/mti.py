# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import scipy.io
import scipy.ndimage

from scilpy.io.image import get_data_as_mask


def py_fspecial_gauss(shape, sigma):
    """
    Function to mimic the 'fspecial gaussian' MATLAB function
    Returns a rotationally symmetric Gaussian lowpass filter of
    shape size with standard deviation sigma.
    see https://www.mathworks.com/help/images/ref/fspecial.html

    Parameters
    ----------
    shape:      Vector to specify the size of square matrix (rows, columns).
                Ex.: (3, 3)
    sigma:      Value for standard deviation (Sigma value of 0.5 is
                recommended).

    Returns
    -------
    Two-dimensional Gaussian filter h of specified size.
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def process_contrast_map(merged_images, single_echo=False,
                         filtering=False):
    """
    Average echoes of a contrast map and apply gaussian filter.

    Parameters
    ----------
    4d_image:       4D image, containing every echoes of the contrast map
    single_echo:    Apply when only one echo is used to compute contrast map
    filtering:      Apply Gaussian filtering to remove Gibbs ringing
                    (default is False).

    Returns
    -------
    Contrast map in 3D-Array.
    """

    if single_echo:
        contrast_map = np.sqrt(np.squeeze(merged_images**2))
    else:
        # Compute the sum of contrast map
        contrast_map = np.sqrt(np.sum(np.squeeze(merged_images**2), 3))

    # Apply gaussian filtering if needed
    if filtering:
        h = py_fspecial_gauss((3, 3), 0.5)
        h = h[:, :, None]
        contrast_map = scipy.ndimage.convolve(contrast_map,
                                              h).astype(np.float32)
    return contrast_map


def compute_saturation_map(MT, PD, T1, a, TR):
    """
    Compute saturation of given contrast (MT).
    Saturation is computed from apparent longitudinal relaxation time
    (T1app) and apparent signal amplitude (Aapp), as the difference of
    longitudinal relaxation times of bound to macromolecules pool
    from free water pool saturation. The estimation of the
    saturation includes correction for the effects of excitation flip angle
    and longitudinal relaxation time, and remove the effect of T1-weighted
    image.

    see Helms et al., 2008
    https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21732

    Parameters
    ----------
    MT:             Contrast of choice (can be a single contrast or the
                    mean of positive and negative for example)
    PD:             Reference proton density weighted image (MToff PD)
    T1:             Reference T1 weighted image (MToff T1)
    a:              List of two flip angles corresponding to PD and T1. If B1
                    correction is on, this should be two 3D-array of flip
                    angles varying spatially with respect to the B1 map.
    TR:             List of two repetition times corresponding to PD and T1.

    Returns
    -------
    MT saturation matrice in 3D-array (sat), computed apparent T1 map (T1app).
    """
    Aapp_num = (TR[0] / (a[0] ** 2)) - (TR[1] / (a[1] ** 2))
    Aapp_den = (TR[0] / (a[0] * PD)) - (TR[1] / (a[1] * T1))
    Aapp = Aapp_num / Aapp_den

    T1app_num = (PD / a[0]) - (T1 / a[1])
    T1app_den = 0.5 * ((T1 * a[1]) / TR[1] - (PD * a[0]) / TR[0])
    T1app = T1app_num / T1app_den

    sat = Aapp * a[0] * TR[0] / T1app / MT - (TR[0] / T1app) - (a[0] ** 2) / 2
    sat *= 100

    return sat, T1app


def compute_ratio_map(mt_on_single, mt_off, mt_on_dual=None):
    """
    Compute magnetization transfer ratio (MTR), and inhomogenous magnetization
    transfer ratio (ihMTR) if mt_on_dual is given.
    - MT ratio (MTR) is computed as the percentage difference of single
    frequency mean image from the reference.
    - ihMT ratio (ihMTR) is computed as the percentage difference of dual from
    single frequency.

    For ihMTR, see Varma et al., 2015
    www.sciencedirect.com/science/article/pii/S1090780715001998

    Parameters
    ----------
    mt_on_single:       MT-on image with single frequency pulse: can be one
                        single positive/negative frequency MT image or the
                        average of both images (positive and negative).
    mt_off:             MT-off image: the reference image without MT
                        preparation.
    mt_on_dual:         MT-on image with dual frequency pulse: can be one
                        dual alternating positive/negative frequency MT image
                        or the average of both images. Optional. If given, will
                        compute the ihMTR also.

    Returns
    -------
    MT ratio (MTR), ihMT ratio (ihMTR).
    """
    # Compute MT Ratio map
    MTR = 100 * ((mt_off - mt_on_single) / mt_off)

    # Compute ihMT ratio map
    if mt_on_dual is not None:
        # The factor 2 is there to account for the /2 in mt_on mean images.
        ihMTR = 2 * 100 * (mt_on_single - mt_on_dual) / mt_off
        return MTR, ihMTR

    return MTR


def threshold_map(computed_map,  in_mask,
                  lower_threshold, upper_threshold,
                  idx_contrast_list=None, contrast_maps=None):
    """
    Remove NaN and apply different threshold based on
    - maximum and minimum threshold value
    - T1 mask
    - combination of specific contrast maps

    idx_contrast_list and contrast_maps are required for
    thresholding of ihMT images.

    Parameters
    ----------
    computed_map:       3D-Array data.
                        Myelin map (ihMT or non-ihMT maps)
    in_mask:            Path to binary T1 mask from T1 segmentation.
                        Must be the sum of GM+WM+CSF.
    lower_threshold:    Value for low thresold <int>
    upper_thresold:     Value for up thresold <int>
    idx_contrast_list:  List of indexes of contrast maps corresponding to
                        that of input contrast_maps ex.: [0, 2, 4]
                        Altnp = 0; Atlpn = 1; Negative = 2; Positive = 3;
                        PD = 4; T1 = 5
    contrast_maps:      List of 3D-Array. File must containing the
                        5 or 6 contrast maps.

    Returns
    -------
    Thresholded matrix in 3D-array.
    """
    # Remove NaN and apply thresold based on lower and upper value
    computed_map[np.isnan(computed_map)] = 0
    computed_map[np.isinf(computed_map)] = 0
    computed_map[computed_map < lower_threshold] = 0
    computed_map[computed_map > upper_threshold] = 0

    # Load and apply sum of T1 probability maps on myelin maps
    if in_mask is not None:
        mask_image = nib.load(in_mask)
        mask_data = get_data_as_mask(mask_image)
        computed_map[np.where(mask_data == 0)] = 0

    # Apply threshold based on combination of specific contrast maps
    if idx_contrast_list and contrast_maps:
        for idx in idx_contrast_list:
            computed_map[contrast_maps[idx] == 0] = 0

    return computed_map


def adjust_B1_map_intensities(B1_map, nominal=100):
    """
    Adjust and verify the values of the B1 map. We want the B1 map to have
    values around 1.0, so we divide by the nominal B1 value if it is not
    already the case. Sometimes the scaling of the B1 map is wrong, leading
    to weird values after this process, which what is verified here.

    Parameters
    ----------
    B1_map:           B1 coregister map.
    nominal:          Nominal values of B1. For Philips, should be 1.

    Returns
    ----------
    Ajusted B1 map in 3d-array.
    """
    B1_map /= nominal
    med_bB = np.nanmedian(np.where(B1_map == 0, np.nan, B1_map))
    if not np.isclose(med_bB, 1.0, atol=0.2):
        raise ValueError("Intensities of the B1 map are wrong.")
    B1_map = np.clip(B1_map, 0.5, 1.5)
    return B1_map


def smooth_B1_map(B1_map, wdims=5):
    """
    Apply a smoothing to the B1 map.

    Parameters
    ----------
    B1_map:           B1 coregister map.
    wdims:            Window dimension (in voxels) for the smoothing.

    Returns
    ----------
    Smoothed B1 map in 3d-array.
    """
    h = np.ones((wdims, wdims, 1)) / wdims ** 2
    B1_map_smooth = scipy.ndimage.convolve(B1_map, h).astype(np.float32)
    return B1_map_smooth


def apply_B1_corr_empiric(MT_map, B1_map):
    """
    Apply an empiric B1 correction on MT map.

    see Weiskopf et al., 2013
    https://www.frontiersin.org/articles/10.3389/fnins.2013.00095/full

    Parameters
    ----------
    MT_map:           3D-Array MT map.
    B1_map:           B1 coregister map.

    Returns
    ----------
    Corrected MT matrix in 3D-array.
    """
    MT_map_B1_corrected = MT_map*(1.0-0.4)/(1-0.4*(B1_map))
    return MT_map_B1_corrected


def apply_B1_corr_model_based(MTsat, B1_map, R1app, fitvalues_file, B1_ref=1):
    """
    Apply a model-based B1 correction on MT map.

    see Rowley et al., 2021
    https://onlinelibrary.wiley.com/doi/10.1002/mrm.28831

    Parameters
    ----------
    MTsat:              3D-Array MTsat map.
    B1_map:             B1 coregister map.
    R1app:              Apparent R1 map, obtained from compute_saturation_map.
    fitvalues_file:     Path to the fitValues_*.mat file corresponding to the
                        MTsat map. This file is computed with Christopher
                        Rowley's Matlab library.
    B1_ref:             Reference B1 value used for the fit (usually 1).

    Returns
    ----------
    Corrected MTsat matrix in 3D-array.
    """
    cf_eq, R1_to_M0b = _read_fitvalues_from_file(fitvalues_file)
    cf_map = _compute_B1_corr_factor_map(B1_map, R1app, cf_eq,
                                         R1_to_M0b, B1_ref=B1_ref)
    MTsat_corr = MTsat + MTsat * cf_map
    return MTsat_corr


def _read_fitvalues_from_file(fitvalues_file):
    """
    Extract equations from fitValues_*.mat file.

    see Rowley et al., 2021
    https://onlinelibrary.wiley.com/doi/10.1002/mrm.28831

    Parameters
    ----------
    fitvalues_file:     Path to the fitValues_*.mat file corresponding to the
                        MTsat map. This file is computed with Christopher
                        Rowley's Matlab library.

    Returns
    ----------
    Correction factor equation (cf_eq), R1 to M0b equation (R1_to_M0b)
    """
    fitvalues = scipy.io.loadmat(fitvalues_file)
    fit_SS_eq = fitvalues['fitValues']['fit_SS_eqn'][0][0][0]
    est_M0b_from_R1 = fitvalues['fitValues']['Est_M0b_from_R1'][0][0][0]
    cf_eq = fit_SS_eq.replace('^',
                              '**').replace('Raobs.',
                                            'Raobs').replace('M0b.',
                                                             'M0b').replace('b1.',
                                                                            'B1')
    R1_to_M0b = est_M0b_from_R1.replace('^',
                                        '**').replace('Raobs.',
                                                      'Raobs').replace('M0b.',
                                                                       'M0b')
    return cf_eq, R1_to_M0b


def _compute_B1_corr_factor_map(B1_map, Raobs, cf_eq, R1_to_M0b, B1_ref=1):
    """
    Compute the correction factor map for B1 correction.

    see Rowley et al., 2021
    https://onlinelibrary.wiley.com/doi/10.1002/mrm.28831

    Parameters
    ----------
    B1_map:             B1 coregister map.
    Raobs:                 R1 map, obtained from compute_saturation_map.
    cf_eq:              Correction factor equation extracted with
                        _read_fitvalues_from_file.
    R1_to_M0b:          Conversion equation from R1 to M0b extracted with
                        _read_fitvalues_from_file.
    B1_ref:             Reference B1 value used for the fit (usually 1).

    Returns
    ----------
    Correction factor map.
    """
    M0b = eval(R1_to_M0b, {"Raobs": Raobs})

    B1 = B1_map * B1_ref
    cf_act = eval(cf_eq, {"Raobs": Raobs, "B1": B1, "M0b": M0b})
    B1 = B1_ref
    cf_nom = eval(cf_eq, {"Raobs": Raobs, "B1": B1, "M0b": M0b})

    cf_map = (cf_nom - cf_act) / cf_act
    return cf_map


def adjust_B1_map_header(B1_img, slope):
    """
    Fixe issue in B1 map header, by applying the scaling (slope) and setting
    the slope to 1.

    Parameters
    ----------
    B1_img: nifti image object
        The B1 map.
    slope: float
        The slope value, obtained from the image header.

    Returns
    ----------
    Adjusted B1 nifti image object (new_b1_img)
    """
    B1_map = B1_img.get_fdata()
    B1_map /= slope
    B1_img.header.set_slope_inter(1, 0)
    new_b1_img = nib.nifti1.Nifti1Image(B1_map, B1_img.affine,
                                        header=B1_img.header)
    return new_b1_img
