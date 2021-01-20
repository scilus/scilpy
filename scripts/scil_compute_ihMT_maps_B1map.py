#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script computes four myelin indices maps from the Magnetization Transfer
(MT) and inhomogeneous Magnetization Transfer (ihMT) images. Magnetization
Transfer is a contrast mechanism in tissue resulting from the proton exchange
between non-aqueous protons (from macromolecules and their closely associated
water molecules, the "bound" pool) and protons in the free water pool called
aqueous protons. This exchange attenuates the MRI signal, introducing
microstructure-dependent contrast. MT's effect reflects the relative density
of macromolecules such as proteins and lipids, it has been associated with
myelin content in white matter of the brain.

Different contrasts can be done with an off-resonance pulse prior to image
acquisition (a prepulse), saturating the protons on non-aqueous molecules,
by applying different frequency irradiation. The two MT maps and two ihMT maps
are obtained using five contrasts: single frequency positive or negative and
dual frequency with an alternation of both positive and negative frequency
(saturated images); and one unsaturated contrast as reference (T1weighted).


Input Data recommendation:
  - it is recommended to use dcm2niix (v1.0.20200331) to convert data
    https://github.com/rordenlab/dcm2niix/releases/tag/v1.0.20200331
  - dcm2niix conversion will create all echo files for each contrast and
    corresponding json files
  - all input must have a matching json file with the same filename
  - all contrasts must have a same number of echoes and coregistered
    between them before running the script.
  - Mask must be coregistered to the echo images
  - ANTs can be used for the registration steps (http://stnava.github.io/ANTs/)


The output consist in two types of images in two folders :
  1. Contrasts_ihMT_maps which contains the 5 contrast images
      - altnp.nii.gz : alternating negative and positive frequency pulses
      - altpn.nii.gz : alternating positive and negative frequency pulses
      - positive.nii.gz : pulses applied at positive frequency
      - negative.nii.gz : pulses applied at negative frequency
      - reference.nii.gz : no pulse

  2. ihMT_native_maps which contains the 4 myelin maps
      - MTR.nii.gz : Magnetization Transfer Ratio map
      - ihMTR.nii.gz : inhomogeneous Magnetization Transfer Ratio map
      The (ih)MT ratio is a measure reflecting the amount of bound protons.

      - MTsat.nii.gz : Magnetization Transfer saturation map
      - ihMTsat.nii.gz : inhomogeneous Magnetization Transfer saturation map
      The (ih)MT saturation is a pseudo-quantitative maps representing
      the signal change between the bound and free water pools.

  These final maps can be corrected by an empiric B1 correction with
  --in_B1_map option, suffix *B1_corrected is added for each map.

>>> scil_compute_ihMT_maps.py path/to/output/directory path/to/mask_bin.nii.gz
    --in_altnp path/to/echo*altnp.nii.gz --in_altpn path/to/echo*altpn.nii.gz
    --in_mtoff path/to/echo*mtoff.nii.gz --in_negative path/to/echo*neg.nii.gz
    --in_positive path/to/echo*pos.nii.gz --in_t1w path/to/echo*T1w.nii.gz

By default, the script uses all the echoes available in the input folder.
If you want to use a single echo add --single_echo to the command line and
replace the * with the specific number of the echo.

"""

import argparse
import os
import json

import nibabel as nib
import numpy as np
import scipy.ndimage

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)

EPILOG = """
Varma G, Girard OM, Prevost VH, Grant AK, Duhamel G, Alsop DC.
Interpretation of magnetization transfer from inhomogeneously broadened lines
(ihMT) in tissues as a dipolar order effect within motion restricted molecules.
Journal of Magnetic Resonance. 1 nov 2015;260:67-76.

Manning AP, Chang KL, MacKay AL, Michal CA. The physical mechanism of
"inhomogeneous" magnetization transfer MRI. Journal of Magnetic Resonance.
1 janv 2017;274:125-36.

Helms G, Dathe H, Kallenberg K, Dechent P. High-resolution maps of
magnetization transfer with inherent correction for RF inhomogeneity
and T1 relaxation obtained from 3D FLASH MRI. Magnetic Resonance in Medicine.
2008;60(6):1396-407.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_dir',
                   help='Path to output folder.')
    p.add_argument('in_mask',
                   help='Path to the T1 binary brain mask. Must be the sum '
                        'of the three tissue probability maps from '
                        'T1 segmentation (GM+WM+CSF).')
    p.add_argument('--out_prefix',
                   help='Prefix to be used for each output image.')
    p.add_argument('--in_B1_map',
                   help='Path to B1 coregister map to MT contrasts.')
    p.add_argument('--filtering', action='store_true',
                   help='Gaussian filtering to remove Gibbs ringing. '
                        'Not recommended.')
    p.add_argument('--single_echo', action='store_true',
                   help='Use this option when there is only one echo.')

    g = p.add_argument_group(title='ihMT contrasts', description='Path to '
                             'echoes corresponding to contrasts images. All '
                             'constrasts must have the same number of echoes '
                             'and coregistered between them. '
                             'Use * to include all echoes.')
    g.add_argument('--in_altnp', nargs='+', required=True,
                   help='Path to all echoes corresponding to the '
                        'alternation of Negative and Positive'
                        'frequency saturation pulse.')
    g.add_argument('--in_altpn', nargs='+', required=True,
                   help='Path to all echoes corresponding to the '
                        'alternation of Positive and Negative '
                        'frequency saturation pulse.')
    g.add_argument("--in_mtoff", nargs='+', required=True,
                   help='Path to all echoes corresponding to the '
                        'no frequency saturation pulse (reference image).')
    g.add_argument("--in_negative", nargs='+', required=True,
                   help='Path to all echoes corresponding to the '
                        'Negative frequency saturation pulse.')
    g.add_argument("--in_positive", nargs='+', required=True,
                   help='Path to all echoes corresponding to the '
                        'Positive frequency saturation pulse.')
    g.add_argument("--in_t1w", nargs='+', required=True,
                   help='Path to all echoes corresponding to the '
                        'T1-weigthed.')

    add_overwrite_arg(p)

    return p


def set_acq_parameters(json_path):
    """
    Function to extract Repetition Time and Flip Angle from json file.

    Parameters
    ----------
    json_path   Path to the json file

    Returns
    ----------
    Return Repetition Time (in second) and Flip Angle (in radians)
    """
    with open(json_path) as f:
        data = json.load(f)
    TR = data['RepetitionTime']*1000
    FlipAngle = data['FlipAngle']*np.pi/180
    return TR, FlipAngle


def merge_images(echoes_image):
    """
    Function to load each echo in a 3D-array matrix and
    concatenate each of them along the 4th dimension.

    Parameters
    ----------
    echoes_image     List : list of echoes path for each contrasts. Ex.
                     ['path/to/echo-1_acq-pos',
                      'path/to/echo-2_acq-pos',
                      'path/to/echo-3_acq-pos']

    Returns
    ----------
    Return a 4D-array matrix of size x, y, z, n where n represented
    the number of echoes.
    """
    merge_array = []
    for echo in range(len(echoes_image)):
        load_image = nib.load(echoes_image[echo])
        merge_array.append(load_image.get_fdata(dtype=np.float32))
    merge_array = np.stack(merge_array, axis=-1)
    return merge_array


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


def compute_contrasts_maps(echoes_image, single_echo=False, filtering=False):
    """
    Load echoes and compute corresponding contrast map.

    Parameters
    ----------
    echoes_image    List of file path : list of echoes path for contrast
    single_echo     Apply when only one echo is used to compute contrast maps
    filtering       Apply Gaussian filtering to remove Gibbs ringing
                    (default is False).

    Returns
    -------
    Contrast map in 3D-Array.
    """

    # Merged the 3 echo images into 4D-array
    merged_map = merge_images(echoes_image)

    if single_echo:
        contrast_map = np.sqrt(np.squeeze(merged_map**2))
    else:
        # Compute the sum of contrast map
        contrast_map = np.sqrt(np.sum(np.squeeze(merged_map**2), 3))

    # Apply gaussian filtering if needed
    if filtering:
        h = py_fspecial_gauss((3, 3), 0.5)
        h = h[:, :, None]
        contrast_map = scipy.ndimage.convolve(contrast_map,
                                              h).astype(np.float32)
    return contrast_map


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
    contrasts_maps:      List of matrices : list of all contrast maps computed
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

    # Compute an ihMTsat image (dR1sat in Varma et al., 2015)
    cPDa = (contrasts_maps[4] + contrasts_maps[3]) / 2
    cPDb = (contrasts_maps[1] + contrasts_maps[0]) / 2
    cT1 = contrasts_maps[5]

    R1appa_num = ((cPDa / acq_parameters[0][1]) - (cT1 / acq_parameters[1][1]))
    R1appa_den = ((cT1*acq_parameters[1][1]) / (2*acq_parameters[1][0] / 1000)
                  - (cPDa*acq_parameters[0][1]) / (2*acq_parameters[0][0]
                                                   / 1000))
    freewater_sat = R1appa_num / R1appa_den

    R1appb_num = ((cPDb / acq_parameters[0][1]) - (cT1 / acq_parameters[1][1]))
    R1appb_den = ((cT1*acq_parameters[1][1]) / (2*acq_parameters[1][0]/1000) -
                  (cPDb*acq_parameters[0][1]) / (2*acq_parameters[0][0]/1000))
    bound_sat = R1appb_num / R1appb_den

    ihMTsat = (1 / bound_sat) - (1 / freewater_sat)

    return ihMTR, ihMTsat


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
               (contrasts_maps[4] + contrasts_maps[3]) / 2)
               / contrasts_maps[2])

    # Compute MT saturation maps
    cPD1 = contrasts_maps[2]
    cPD2 = (contrasts_maps[4] + contrasts_maps[3]) / 2
    cT1 = contrasts_maps[5]

    Aapp_num = ((2*acq_parameters[0][0] / (acq_parameters[0][1]**2)) -
                (2*acq_parameters[1][0] / (acq_parameters[1][1]**2)))
    Aapp_den = (((2*acq_parameters[0][0]) / (acq_parameters[0][1]*cPD1)) -
                ((2*acq_parameters[1][0]) / (acq_parameters[1][1]*cT1)))
    Aapp = Aapp_num / Aapp_den

    R1app_num = ((cPD1 / acq_parameters[0][1]) - (cT1 / acq_parameters[1][1]))
    R1app_den = ((cT1*acq_parameters[1][1]) / (2*acq_parameters[1][0]) -
                 (cPD1*acq_parameters[0][1]) / (2*acq_parameters[0][0]))
    R1app = R1app_num / R1app_den

    MTsat = 100*(((Aapp*acq_parameters[0][1]*acq_parameters[0][0] / R1app)
                 / cPD2) - (acq_parameters[0][0] / R1app)
                 - (acq_parameters[0][1]**2) / 2)

    return MTR, MTsat


def threshold_ihMT_maps(computed_map, contrasts_maps, in_mask,
                        lower_threshold, upper_threshold, idx_contrast_list):
    """
    Remove NaN and apply different threshold based on
       - maximum and minimum threshold value
       - T1 mask
       - combination of specific contrasts maps

    Parameters
    ----------
    computed_map        3D-Array data.
                        Myelin map (ihMT or non-ihMT maps)
    contrasts_maps      List of 3D-Array. File must containing the
                        6 contrasts maps.
    in_mask             Path to binary T1 mask from T1 segmentation.
                        Must be the sum of GM+WM+CSF.
    lower_threshold     Value for low thresold <int>
    upper_thresold      Value for up thresold <int>
    idx_contrast_list   List of indexes of contrast maps corresponding to
                        that of input contrasts_maps ex.: [0, 2, 5]
                        Altnp = 0; Atlpn = 1; Reference = 2; Negative = 3;
                        Positive = 4; T1weighted = 5
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
    for idx in idx_contrast_list:
        computed_map[contrasts_maps[idx] == 0] = 0

    return computed_map


def apply_B1_correction(MT_map, B1_map):
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
    # Load B1 image
    B1_img = nib.load(B1_map)
    B1_img_data = B1_img.get_fdata(dtype=np.float32)

    # Apply a light smoothing to the B1 map
    h = np.ones((5, 5, 1))/25
    B1_smooth_map = scipy.ndimage.convolve(B1_img_data,
                                           h).astype(np.float32)

    # Apply an empiric B1 correction via B1 smooth data on MT data
    MT_map_B1_corrected = MT_map*(1.0-0.4)/(1-0.4*(B1_smooth_map/100))

    return MT_map_B1_corrected


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_output_dirs_exist_and_empty(parser, args,
                                       os.path.join(args.out_dir,
                                                    'Contrasts_ihMT_maps'),
                                       os.path.join(args.out_dir,
                                                    'ihMT_native_maps'),
                                       create_dir=True)

    # Merge all echos path into a list
    maps = [args.in_altnp, args.in_altpn, args.in_mtoff, args.in_negative,
            args.in_positive, args.in_t1w]

    maps_flat = (args.in_altnp + args.in_altpn + args.in_mtoff +
                 args.in_negative + args.in_positive + args.in_t1w)

    jsons = [curr_map.replace('.nii.gz', '.json')
             for curr_map in maps_flat]

    # check echoes number and jsons
    assert_inputs_exist(parser, jsons + maps_flat)
    for curr_map in maps[1:]:
        if len(curr_map) != len(maps[0]):
            parser.error('Not the same number of echoes per contrast')

    # Set TR and FlipAngle parameters for ihMT (positive contrast)
    # and T1w images
    parameters = [set_acq_parameters(maps[4][0].replace('.nii.gz', '.json')),
                  set_acq_parameters(maps[5][0].replace('.nii.gz', '.json'))]

    # Fix issue from the presence of invalide value and division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Define reference image for saving maps
    ref_img = nib.load(maps[4][0])

    # Define contrasts maps names
    contrasts_name = ['altnp', 'altpn', 'reference', 'negative', 'positive',
                      'T1w']
    if args.filtering:
        contrasts_name = [curr_name + '_filter'
                          for curr_name in contrasts_name]
    if args.single_echo:
        contrasts_name = [curr_name + '_single_echo'
                          for curr_name in contrasts_name]

    if args.out_prefix:
        contrasts_name = [args.out_prefix + '_' + curr_name
                          for curr_name in contrasts_name]

    # Compute contrasts maps
    computed_contrasts = []
    for idx, curr_map in enumerate(maps):
        computed_contrasts.append(compute_contrasts_maps(
                                  curr_map, filtering=args.filtering,
                                  single_echo=args.single_echo))

        nib.save(nib.Nifti1Image(computed_contrasts[idx].astype(np.float32),
                                 ref_img.affine),
                 os.path.join(args.out_dir, 'Contrasts_ihMT_maps',
                              contrasts_name[idx] + '.nii.gz'))

    # Compute and thresold ihMT maps
    ihMTR, ihMTsat = compute_ihMT_maps(computed_contrasts, parameters)
    ihMTR = threshold_ihMT_maps(ihMTR, computed_contrasts, args.in_mask,
                                0, 100, [4, 3, 1, 0, 2])
    ihMTsat = threshold_ihMT_maps(ihMTsat, computed_contrasts, args.in_mask,
                                  0, 10, [4, 3, 1, 0])
    if args.in_B1_map:
        ihMTR = apply_B1_correction(ihMTR, args.in_B1_map)
        ihMTsat = apply_B1_correction(ihMTsat, args.in_B1_map)

    # Compute and thresold non-ihMT maps
    MTR, MTsat = compute_MT_maps(computed_contrasts, parameters)
    for curr_map in MTR, MTsat:
        curr_map = threshold_ihMT_maps(curr_map, computed_contrasts,
                                       args.in_mask, 0, 100, [4, 2])
        if args.in_B1_map:
            curr_map = apply_B1_correction(curr_map, args.in_B1_map)

    # Save ihMT and MT images
    img_name = ['ihMTR', 'ihMTsat', 'MTR', 'MTsat']

    if args.filtering:
        img_name = [curr_name + '_filter'
                    for curr_name in img_name]

    if args.single_echo:
        img_name = [curr_name + '_single_echo'
                    for curr_name in img_name]

    if args.in_B1_map:
        img_name = [curr_name + '_B1_corrected'
                    for curr_name in img_name]

    if args.out_prefix:
        img_name = [args.out_prefix + '_' + curr_name
                    for curr_name in img_name]

    img_data = ihMTR, ihMTsat, MTR, MTsat
    for img_to_save, name in zip(img_data, img_name):
        nib.save(nib.Nifti1Image(img_to_save.astype(np.float32),
                                 ref_img.affine),
                 os.path.join(args.out_dir, 'ihMT_native_maps',
                              name + '.nii.gz'))


if __name__ == '__main__':
    main()
