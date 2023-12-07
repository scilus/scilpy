# -*- coding: utf-8 -*-

import json

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
    shape    Vector to specify the size of square matrix (rows, columns).
             Ex.: (3, 3)
    sigma    Value for standard deviation (Sigma value of 0.5 is recommended).

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


def compute_contrasts_maps(merged_images, single_echo=False,
                           filtering=False):
    """
    Load echoes and compute corresponding contrast map.

    Parameters
    ----------
    4d_image        4D images
    single_echo     Apply when only one echo is used to compute contrast maps
    filtering       Apply Gaussian filtering to remove Gibbs ringing
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


def compute_saturation(cPD1, cPD2, cT1, acq_parameters):
    """
    Compute saturation of given contrast.
    Saturation is computed from apparent longitudinal relaxation rate
    (R1app) and apparent signal amplitude (Aapp). The estimation of the
    saturation includes correction for the effects of excitation flip angle
    and longitudinal relaxation rate, and remove the effect of T1-weighted
    image.

    see Helms et al., 2008
    https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21732

    Parameters
    ----------
    cPD1:               Reference proton density (MT-off)
    cPD2:               Contrast of choice (can be a single contrast or the
                        mean of positive and negative for example)
    cT1:                Contrast T1-weighted
    acq_parameters:     List of TR and Flipangle for MT contrast and T1w images
                        [TR, Flipangle]
    Returns
    -------
    MT ratio and MT saturation matrice in 3D-array.
    """
    Aapp_num = ((2*acq_parameters[0][0] / (acq_parameters[0][1]**2)) -
                (2*acq_parameters[1][0] / (acq_parameters[1][1]**2)))
    Aapp_den = (((2*acq_parameters[0][0]) / (acq_parameters[0][1]*cPD1)) -
                ((2*acq_parameters[1][0]) / (acq_parameters[1][1]*cT1)))
    Aapp = Aapp_num / Aapp_den

    R1app_num = ((cPD1 / acq_parameters[0][1]) - (cT1 / acq_parameters[1][1]))
    R1app_den = ((cT1*acq_parameters[1][1]) / (2*acq_parameters[1][0]) -
                 (cPD1*acq_parameters[0][1]) / (2*acq_parameters[0][0]))
    R1app = R1app_num / R1app_den

    sat = 100*(((Aapp*acq_parameters[0][1]*acq_parameters[0][0] / R1app
                 ) / cPD2) - (acq_parameters[0][0] / R1app) -
               (acq_parameters[0][1]**2) / 2)

    return sat


def compute_ihMT_maps(contrasts_maps, acq_parameters):
    """
    Compute Inhomogenous Magnetization transfer ratio and saturation maps.
    - ihMT ratio (ihMTR) is computed as the percentage difference of dual from
    single frequency.
    - ihMT saturation (ihMTsat) is computed as the difference of longitudinal
    relaxation rates of bound to macromolecules pool from free water pool
    saturation. The estimation of the ihMT saturation includes correction for
    the effects of excitation flip angle and longitudinal relaxation rate, and
    remove the effect of T1-weighted image.
        cPD : contrast proton density
            a : mean of positive and negative proton density
            b : mean of two dual proton density
        cT1 : contrast T1-weighted
        num : numberator
        den : denumerator

    see Varma et al., 2015
    www.sciencedirect.com/science/article/pii/S1090780715001998
    see Manning et al., 2017
    www.sciencedirect.com/science/article/pii/S1090780716302488?via%3Dihub

    Parameters
    ----------
    contrasts_maps:     List of matrices : list of all contrast maps computed
                                           with compute_contrasts_maps
    acq_parameters:     List of TR and Flipangle values for ihMT and T1w images
                        [TR, Flipangle]
    Returns
    -------
    ihMT ratio (ihMTR) and ihMT saturation (ihMTsat) matrices in 3D-array.

    """
    # Compute ihMT ratio map
    ihMTR = 100*(contrasts_maps[4] + contrasts_maps[3] -
                 contrasts_maps[1] - contrasts_maps[0]) / contrasts_maps[2]

    # Compute MT saturation maps
    cPD1 = contrasts_maps[2]
    cPDa = (contrasts_maps[4] + contrasts_maps[3]) / 2
    cPDb = (contrasts_maps[1] + contrasts_maps[0]) / 2
    cT1 = contrasts_maps[5]

    MT_sat_single = compute_saturation(cPD1, cPDa, cT1, acq_parameters)
    MT_sat_dual = compute_saturation(cPD1, cPDb, cT1, acq_parameters)
    ihMTsat = MT_sat_dual - MT_sat_single

    return ihMTR, ihMTsat


def compute_MT_maps_from_ihMT(contrasts_maps, acq_parameters):
    """
    Compute Magnetization transfer ratio and saturation maps.
    MT ratio is computed as the percentage difference of two images, one
    acquired with off-resonance saturation (MT-on) and one without (MT-off).
    MT saturation is computed from apparent longitudinal relaxation rate
    (R1app) and apparent signal amplitude (Aapp). The estimation of the MT
    saturation includes correction for the effects of excitation flip angle
    and longitudinal relaxation rate, and remove the effect of T1-weighted
    image.
        cPD : contrast proton density
            1 : reference proton density (MT-off)
            2 : mean of positive and negative proton density (MT-on)
        cT1 : contrast T1-weighted
        num : numberator
        den : denumerator

    see Helms et al., 2008
    https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21732

    Parameters
    ----------
    contrasts_maps:     List of 3D-array constrats matrices : list of all
                        contrast maps computed with compute_ihMT_contrasts
    acq_parameters:     List of TR and Flipangle for ihMT and T1w images
                        [TR, Flipangle]
    Returns
    -------
    MT ratio and MT saturation matrice in 3D-array.
    """
    # Compute MT Ratio map
    MTR = 100*((contrasts_maps[2] -
               (contrasts_maps[4] + contrasts_maps[3]) / 2) /
               contrasts_maps[2])

    # Compute MT saturation maps
    cPD1 = contrasts_maps[2]
    cPD2 = (contrasts_maps[4] + contrasts_maps[3]) / 2
    cT1 = contrasts_maps[5]

    MTsat = compute_saturation(cPD1, cPD2, cT1, acq_parameters)

    return MTR, MTsat


def compute_MT_maps(contrasts_maps, acq_parameters):
    """
    Compute Magnetization transfer ratio and saturation maps.
    MT ratio is computed as the percentage difference of two images, one
    acquired with off-resonance saturation (MT-on) and one without (MT-off).
    MT saturation is computed from apparent longitudinal relaxation rate
    (R1app) and apparent signal amplitude (Aapp). The estimation of the MT
    saturation includes correction for the effects of excitation flip angle
    and longitudinal relaxation rate, and remove the effect of T1-weighted
    image.
        cPD : contrast proton density
            1 : reference proton density (MT-off)
            2 : mean of positive and negative proton density (MT-on)
        cT1 : contrast T1-weighted
        num : numberator
        den : denumerator

    see Helms et al., 2008
    https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21732

    Parameters
    ----------
    contrasts_maps:      List of 3D-array constrats matrices : list of all
                        contrast maps computed with compute_ihMT_contrasts
    acq_parameters:     List of TR and Flipangle for ihMT and T1w images
                        [TR, Flipangle]
    Returns
    -------
    MT ratio and MT saturation matrice in 3D-array.
    """
    # Compute MT Ratio map
    MTR = 100*(contrasts_maps[0] - contrasts_maps[1]) / contrasts_maps[0]

    # Compute MT saturation maps: cPD1 = mt-off; cPD2 = mt-on
    cPD1 = contrasts_maps[0]
    cPD2 = contrasts_maps[1]
    cT1 = contrasts_maps[2]

    Aapp_num = ((2*acq_parameters[0][0] / (acq_parameters[0][1]**2)) -
                (2*acq_parameters[1][0] / (acq_parameters[1][1]**2)))
    Aapp_den = (((2*acq_parameters[0][0]) / (acq_parameters[0][1]*cPD1)) -
                ((2*acq_parameters[1][0]) / (acq_parameters[1][1]*cT1)))
    Aapp = Aapp_num / Aapp_den

    R1app_num = ((cPD1 / acq_parameters[0][1]) - (cT1 / acq_parameters[1][1]))
    R1app_den = ((cT1*acq_parameters[1][1]) / (2*acq_parameters[1][0]) -
                 (cPD1*acq_parameters[0][1]) / (2*acq_parameters[0][0]))
    R1app = R1app_num / R1app_den

    MTsat = 100*(((Aapp*acq_parameters[0][1]*acq_parameters[0][0]/R1app) /
                  cPD2) - (acq_parameters[0][0]/R1app) -
                 (acq_parameters[0][1]**2)/2)

    return MTR, MTsat


def threshold_maps(computed_map,  in_mask,
                   lower_threshold, upper_threshold,
                   idx_contrast_list=None, contrasts_maps=None):
    """
    Remove NaN and apply different threshold based on
       - maximum and minimum threshold value
       - T1 mask
       - combination of specific contrasts maps
    idx_contrast_list and contrasts_maps are required for
    thresholding of ihMT images.

    Parameters
    ----------
    computed_map        3D-Array data.
                        Myelin map (ihMT or non-ihMT maps)
    in_mask             Path to binary T1 mask from T1 segmentation.
                        Must be the sum of GM+WM+CSF.
    lower_threshold     Value for low thresold <int>
    upper_thresold      Value for up thresold <int>
    idx_contrast_list   List of indexes of contrast maps corresponding to
                        that of input contrasts_maps ex.: [0, 2, 5]
                        Altnp = 0; Atlpn = 1; Reference = 2; Negative = 3;
                        Positive = 4; T1weighted = 5
    contrasts_maps      List of 3D-Array. File must containing the
                        6 contrasts maps.
    Returns
    ----------
    Thresholded matrix in 3D-array.
    """
    # Remove NaN and apply thresold based on lower and upper value
    computed_map[np.isnan(computed_map)] = 0
    computed_map[np.isinf(computed_map)] = 0
    computed_map[computed_map < lower_threshold] = 0
    computed_map[computed_map > upper_threshold] = 0

    # Load and apply sum of T1 probability maps on myelin maps
    mask_image = nib.load(in_mask)
    mask_data = get_data_as_mask(mask_image)
    computed_map[np.where(mask_data == 0)] = 0

    # Apply threshold based on combination of specific contrasts maps
    if idx_contrast_list and contrasts_maps:
        for idx in idx_contrast_list:
            computed_map[contrasts_maps[idx] == 0] = 0

    return computed_map


def apply_B1_correction_empiric(MT_map, B1_map):
    """
    Function to apply an empiric B1 correction.

    see Weiskopf et al., 2013
    https://www.frontiersin.org/articles/10.3389/fnins.2013.00095/full

    Parameters
    ----------
    MT_map           3D-Array Myelin map.
    B1_map           Path to B1 coregister map.

    Returns
    ----------
    Corrected MT matrix in 3D-array.
    """

    # Apply an empiric B1 correction via B1 smooth data on MT data
    MT_map_B1_corrected = MT_map*(1.0-0.4)/(1-0.4*(B1_map))

    return MT_map_B1_corrected


def ajust_b1_map_intensities(b1_map, nominal=100):
    b1_map /= nominal
    mean_b1 = np.mean(b1_map)
    if np.isclose(mean_b1, 1.0, atol=0.2):
        raise ValueError("Intensities of the B1 map are wrong.")
    b1_map = np.clip(b1_map, 0.5, 1.4) # Not sure if should clip or cut
    return b1_map


def smooth_B1_map(B1_map):
    # Apply a light smoothing to the B1 map
    h = np.ones((5, 5, 1))/25
    B1_map_smooth = scipy.ndimage.convolve(B1_map, h).astype(np.float32)
    return B1_map_smooth


def read_fit_values_from_mat_file(fit_values_file):
    fit_values = scipy.io.loadmat(fit_values_file)
    coeffs = fit_values['fitValues']['fitvals_coeff'][0][0][0]
    fit_SS_eqn = fit_values['fitValues']['fit_SS_eqn'][0][0][0]
    Est_M0b_from_R1 = fit_values['fitValues']['Est_M0b_from_R1'][0][0][0]
    cf_eq = fit_SS_eqn.replace('^', '**').replace('Raobs.','Raobs').replace('M0b.','M0b').replace('b1.','b1')
    r1_to_m0b = Est_M0b_from_R1.replace('^', '**').replace('Raobs.','Raobs').replace('M0b.','M0b').replace('b1.','b1')
    return coeffs


def compute_B1_correction_factor_map(b1_map, r1, coeffs, r1_to_m0b, b1_ref=1):
    b1_coeff_num = 5
    m0b_coeff_num = 2
    r1_coeff_num = 4

    # low_flip_angle = 15;    % flip angle in degrees -> Customize
    # high_flip_angle = 30;  % flip angle in degrees -> Customize
    # TR1 = 112;               % low flip angle repetition time of the GRE kernel in milliseconds -> Customize
    # TR2 = 20;               % high flip angle repetition time of the GRE kernel in milliseconds -> Customize

    # a1 = deg2rad(low_flip_angle) * b1; 
    # a2 = deg2rad(high_flip_angle)* b1; 

    # R1 = 0.5 .* (hfa.*a2./ TR2 - lfa.*a1./TR1) ./ (lfa./(a1) - hfa./(a2));
    # App = lfa .* hfa .* (TR1 .* a2./a1 - TR2.* a1./a2) ./ (hfa.* TR1 .*a2 - lfa.* TR2 .*a1);

    # T1 = 1./R1 .*mask1;
    # R1_s = (1./T1) *1000; % convert to 1/ms to 1/s

    m0b = r1_to_m0b(r1)

    b1 = b1_map * b1_ref

    # Try eval() instead.
    # cf_act = eval(cf_eq)

    idx = 0
    cf_act = np.zeros(b1_map.shape)
    cf_nom = np.zeros(b1_map.shape)
    for i in range(0, m0b_coeff_num + 1):
        for j in range(0, b1_coeff_num + 1):
            for k in range(0, r1_coeff_num + 1):
                cf_act += coeffs[idx] * m0b ** i * b1 ** j * r1 ** k
                cf_nom += coeffs[idx] * m0b ** i * b1_ref ** j * r1 ** k

                idx += 1

    cf_map = (cf_nom - cf_act) / cf_act

    return cf_map


def apply_B1_correction_model_based(MTsat, cf_map):
    MTsat_corr = MTsat + MTsat * cf_map
    return MTsat_corr