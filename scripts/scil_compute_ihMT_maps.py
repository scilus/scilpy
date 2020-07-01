# -*- coding: utf-8 -*-
"""
Computing ihMT and non-ihMT maps
Make a more details description !
"""

import argparse
import os

import glob
import json
import nibabel as nib
import numpy as np
import scipy.ndimage

from scilpy.io.utils import (add_overwrite_arg,
                             assert_output_dirs_exist_and_empty)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_folder",
                   help='Path to the subject ihMT maps folder')
    p.add_argument('in_mask',
                   help='Path to subject T1 probability mask'
                        'Must be the sum of 3 (WM+GM+CSF) brain compartments.')
    p.add_argument('out_dir',
                   help='Path to subject output folder')
    p.add_argument('--filtering', default=None,
                   help='Gaussian filtering to remove Gibbs ringing.'
                        'Not recommanded.')

    add_overwrite_arg(p)

    return p


def set_acq_parameters(json_path):
    """
    Function to extract Repetition Time and Flip Angle.

    Parameters
    ----------
    json_path          Path to the json file

    Returns
    ----------
    Return Repetition Time (in second) and Flip Angle (in radians)
    """
    with open(json_path) as f:
        data = json.load(f)
    RT = data['RepetitionTime']*1000
    FlipAngle = data['FlipAngle']*np.pi/180
    return RT, FlipAngle


def merge_images(echo_images):
    """
    Function to load each echos images into list of 3D-array matrix and
    concatenate each of them along the 4th dimension.

    Parameters
    ----------
    in_maps          List of images path : list of echos path for each
                     contrasts in subject folder. Ex.
                     ['path/to/echo-1_acq-pos',
                      'path/to/echo-2_acq-pos',
                      'path/to/echo-3_acq-pos']

    Returns
    ----------
    Return a list of 4D-array (x, y, z, n) matrix of n 3D-array (x, y, z)
    matrices as input. N represented the number of echo.
    """
    merge_array = []
    for echo in range(len(echo_images)):
        load_image = nib.load(echo_images[echo])
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
    sigma    Value for standard deviation (Sigma value of 0.5 is recommanded).

    Returns
    -------
    Return two-dimensional Gaussian filter h of specified size.
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def compute_contrasts_maps(echo_images, filtering=False):
    """
    Load echo images and compute contrasts maps : more description details !

    Parameters
    ----------
    echo_images     List of file path : list of echo(s) path for each
                    contrasts
    filtering       Apply Gaussian filtering to remove Gibbs ringing
                    (default is None).

    Returns
    -------
    Returns contrast map in 3D-Array.
    """

    # Merged the 3 echo images into 4D-array
    merged_map = merge_images(echo_images)

    # Compute the sum of contrast map
    contrast_map = np.sqrt(np.sum(np.squeeze(merged_map**2), 3))

    # Apply gaussian filtering if needed
    if filtering:
        h = py_fspecial_gauss((3, 3), 0.5)
        h = h[:, :, None]
        contrast_map = scipy.ndimage.convolve(contrast_map, h).astype(np.float32)
    return contrast_map


def compute_ihMT_maps(contrast_maps, acq_parameters):
    """
    Compute ihMT maps : More description details coming soon !
    Use of both single frequency irradiation positive and negative,
    compensates efficiently for MT asymmetry (finir de modifier)

    see Varma et al., 2015
    https://www.sciencedirect.com/science/article/pii/S1090780715001998

    Parameters
    ----------
    contrast_maps:      List of constrats : list of all contrast maps computed
                                            with compute_ihMT_contrasts
    acq_parameters:     Lists of RT and Flipangle for ihMT and T1w
                        [RT, Flipangle]
    Returns
    -------
    ihMT maps in 3D-Array : ihMT ratio and saturation Maps.

    """
    # Compute ihMT ratio map
    ihMTR = 100*(contrast_maps[4]+contrast_maps[3]-contrast_maps[1] - contrast_maps[0])/contrast_maps[2]

# flake8 to fix: longueur ligne jamais content ! Je ne sais pas où "couper"
    # Compute an dR1sat image
    cPD1a = (contrast_maps[4]+contrast_maps[3])/2
    cPD1b = (contrast_maps[1]+contrast_maps[0])/2
    cT1 = contrast_maps[5]
    T1appa = ((cPD1a/acq_parameters[0][1])-(cT1/acq_parameters[1][1]))/((cT1*acq_parameters[1][1])/(2*acq_parameters[1][0]/1000)-(cPD1a*acq_parameters[0][1])/(2*acq_parameters[0][0]/1000))
    T1appb = ((cPD1b/acq_parameters[0][1])-(cT1/acq_parameters[1][1]))/((cT1*acq_parameters[1][1])/(2*acq_parameters[1][0]/1000)-(cPD1b*acq_parameters[0][1])/(2*acq_parameters[0][0]/1000))
    ihMTsat = (1/T1appb)-(1/T1appa)

    return ihMTR, ihMTsat


def compute_MT_maps(contrast_maps, acq_parameters):
    """
    Compute MT maps : description details coming soon !

    Parameters
    ----------
    contrast_maps:      List of constrats matrices : list of all contrast maps
                        computed with compute_ihMT_contrasts
    acq_parameters:     Lists of RT and Flipangle for ihMT and T1w
                        [RT, Flipangle]
    Returns
    -------
    Non-ihMT maps in 3D-Array : MT ratio and saturation Maps.
    """
    # Compute MT Ratio map
    MTR = 100*((contrast_maps[2]-(contrast_maps[4]+contrast_maps[3])/2)/contrast_maps[2])

# flake8 to fix: meme rangaine longueur ligne où couper ???
    # Compute MT sat maps
    cPD1 = contrast_maps[2]
    cPD2 = (contrast_maps[4]+contrast_maps[3])/2
    cT1 = contrast_maps[5]
    Aapp = ((2*acq_parameters[0][0]/(acq_parameters[0][1]**2))-(2*acq_parameters[1][0]/(acq_parameters[1][1]**2)))/(((2*acq_parameters[0][0])/(acq_parameters[0][1]*cPD1))-((2*acq_parameters[1][0])/(acq_parameters[1][1]*cT1)))
    T1app = ((cPD1/acq_parameters[0][1])-(cT1/acq_parameters[1][1]))/((cT1*acq_parameters[1][1])/(2*acq_parameters[1][0])-(cPD1*acq_parameters[0][1])/(2*acq_parameters[0][0]))
    MTsat = 100*(((Aapp*acq_parameters[0][1]*acq_parameters[0][0]/T1app)/cPD2)-(acq_parameters[0][0]/T1app)-(acq_parameters[0][1]**2)/2)

    return MTR, MTsat


def threshold_ihMT_maps(computed_map, contrast_maps, in_mask, lower_threshold,
                        upper_threshold, idx_contrast_list):
    """
    Remove NaN, apply different threshold based on
       - maximum and minimum threshold value
       - combination of specific contrasts maps
    Multiple data by probability T1 mask (GM+WM+CSF)

    Parameters
    ----------
    computed_map        3D-Array data.
                        Myelin map (ihMT or non-ihMT computed maps)
    contrast_maps       List of 3D-Array. File must containing the
                        5 contrasts maps.
    in_mask             Path to probability T1 map
    lower_threshold     Value for low thresold <int>
    upper_thresold      Value for up thresold <int>
    idx_contrast_list   List of indice contrasts maps in order corresponding to
                        contrast_maps input ex.: [1, 2, 3]
                        Altnp = 0; Atlpn = 1; Reference = 2; Negative = 3;
                        Positive = 4; T1weighted = 5
    Returns
    ----------
    Thresholded matrice in 3D-array.
    """
    # Remove NaN and apply thresold based on lower and upper value
    computed_map[np.isnan(computed_map)] = 0
    computed_map[np.isinf(computed_map)] = 0
    computed_map[computed_map < lower_threshold] = 0
    computed_map[computed_map > upper_threshold] = 0

    # Load and apply probability masks on maps
    mask_image = nib.load(in_mask)
    mask_data = mask_image.get_fdata(dtype=np.float32)
    computed_map[np.where(mask_data == 0)] = 0

    # Apply threshold based on combination of specific contrasts maps
    for idx in idx_contrast_list:
        computed_map[contrast_maps[idx] == 0] = 0

    return computed_map


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_output_dirs_exist_and_empty(parser, args, os.path.join(args.out_dir, 'Contrats_MT_maps'),
                                       os.path.join(args.out_dir, 'ihMT_native_maps'),
                                       create_dir=True)

    # Select contrasts files from input folder
    maps = []
    acquisitions = ['altnp', 'altpn', 'mtoff', 'neg', 'pos', 'T1w']
    for acquisition in acquisitions:
        maps.append(sorted(glob.glob(os.path.join(args.in_folder,
                                                  '*' + acquisition + '*gz'))))

    # Set RT and FlipAngle parameters for ihMT and T1w images
    parameters = [set_acq_parameters(maps[4][0].replace('_warped.nii.gz', '.json')),
                  set_acq_parameters(maps[5][0].replace('_warped.nii.gz', '.json'))]

    # Fix issue from the presence of NaN into array
    np.seterr(divide='ignore', invalid='ignore')

    # Define reference image for saving maps
    ref_img = nib.load(maps[4][0])

    # Define contrasts maps names
    if args.filtering:
        contrasts_name = ['altnp_filter', 'altpn_filter', 'reference_filter',
                          'negative_filter', 'positive_filter', 'T1w_filter']
    else:
        contrasts_name = ['altnp', 'altpn', 'reference', 'negative', 'positive',
                          'T1w']

    # Compute contrasts maps
    computed_contrasts = []
    for idx, curr_map in enumerate(maps):
        computed_contrasts.append(compute_contrasts_maps(curr_map,
                                                         filtering=args.filtering))

        nib.save(nib.Nifti1Image(computed_contrasts[idx].astype(np.float64),
                                 ref_img.affine, ref_img.header),
                 os.path.join(args.out_dir, 'Contrats_MT_maps',
                              contrasts_name[idx]))

    # Compute and thresold ihMT maps
    ihMTR, ihMTsat = compute_ihMT_maps(computed_contrasts, parameters)
    ihMTR = threshold_ihMT_maps(ihMTR, computed_contrasts, args.in_mask, 0, 100,
                                [4, 3, 1, 0, 2])
    ihMTsat = threshold_ihMT_maps(ihMTsat, computed_contrasts, args.in_mask,
                                  0, 10, [4, 3, 1, 0])

    # Compute and thresold non-ihMT maps
    MTR, MTsat = compute_MT_maps(computed_contrasts, parameters)
    for curr_map in MTR, MTsat:
        curr_map = threshold_ihMT_maps(curr_map, computed_contrasts,
                                       args.in_mask, 0, 100, [4, 2])

    # Save ihMT and MT images as Nifti format
    if args.filtering:
        img_name = ['ihMTR_filter', 'ihMTsat_filter', 'MTR_filter',
                    'MTsat_filter']
    else:
        img_name = ['ihMTR', 'ihMTsat', 'MTR', 'MTsat']

    img_data = ihMTR, ihMTsat, MTR, MTsat
    for img_to_save, name in zip(img_data, img_name):
        nib.save(nib.Nifti1Image(img_to_save.astype(np.float64),
                                 ref_img.affine, ref_img.header),
                 os.path.join(args.out_dir, 'ihMT_native_maps', name))


if __name__ == '__main__':
    main()
