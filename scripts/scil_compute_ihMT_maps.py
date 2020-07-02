# -*- coding: utf-8 -*-
"""
Computing ihMT and non-ihMT maps
Make a more details description !
"""

import argparse
import os

import json
import nibabel as nib
import numpy as np
import scipy.ndimage

from scilpy.io.utils import (add_overwrite_arg,
                             assert_output_dirs_exist_and_empty)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('id_subj',
                   help='Subject name for saving maps.')
    p.add_argument('out_dir',
                   help='Path to output folder.')
    p.add_argument('in_mask', nargs='+',
                   help='Path to the probability brain mask. Must be the sum '
                        'of the three tissue probability maps '
                        'from T1 segmentation (GM+WM+CSF).')
    p.add_argument('--filtering', action_store=True,
                   help='Gaussian filtering to remove Gibbs ringing.'
                        'Not recommanded.')

    g = p.add_argument_group(title='ihMT contrasts', description='Path to '
                             'echoes corresponding to contrasts images. All '
                             'constrat must have the same number of echos.')
    g.add_argument('--in_altnp', nargs='+',
                   help='Path to all echoes corresponding to altnp images '
                        'alternation of Negative and Positive'
                        'frequency saturation pulse.')
    g.add_argument('--in_altpn', nargs='+',
                   help='Path to all echoes corresponding to the '
                        'alternation of Positive and Negative '
                        'frequency saturation pulse.')
    g.add_argument("--in_mtoff", nargs='+',
                   help='Path to all echoes corresponding to the '
                        'no frequency saturation pulse (reference image).')
    g.add_argument("--in_negative", nargs='+',
                   help='Path to all echoes corresponding to the '
                        'Negative frequency saturation pulse.')
    g.add_argument("--in_positive", nargs='+',
                   help='Path to all echoes corresponding to the '
                        'Positive frequency saturation pulse.')
    g.add_argument("--in_t1w", nargs='+',
                   help='Path to all echoes corresponding to the '
                        'T1weigthed images.')

    add_overwrite_arg(p)

    return p


def set_acq_parameters(json_path):
    """
    Function to extract Repetition Time and Flip Angle from json file.

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


def merge_images(echoes_image):
    """
    Function to load each echo in a 3D-array matrix and
    concatenate each of them along the 4th dimension.

    Parameters
    ----------
    echo_images      List : list of echoes path for each contrasts. Ex.
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
    sigma    Value for standard deviation (Sigma value of 0.5 is recommanded).

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


def compute_contrasts_maps(echoes_image, filtering=False):
    """
    Load echoes and calculate contrast maps : more description details soon !

    Parameters
    ----------
    echo_images     List of file path : list of echo(s) path for each
                    contrasts
    filtering       Apply Gaussian filtering to remove Gibbs ringing
                    (default is None).

    Returns
    -------
    Contrast map in 3D-Array.
    """

    # Merged the 3 echo images into 4D-array
    merged_map = merge_images(echoes_image)

    # Compute the sum of contrast map
    contrast_map = np.sqrt(np.sum(np.squeeze(merged_map**2), 3))

    # Apply gaussian filtering if needed
    if filtering:
        h = py_fspecial_gauss((3, 3), 0.5)
        h = h[:, :, None]
        contrast_map = scipy.ndimage.convolve(contrast_map,
                                              h).astype(np.float32)
    return contrast_map


def compute_ihMT_maps(contrast_maps, acq_parameters):
    """
    Compute ihMT maps : More description details coming soon !

    see Varma et al., 2015
    https://www.sciencedirect.com/science/article/pii/S1090780715001998

    Parameters
    ----------
    contrast_maps:      List of constrats : list of all contrast maps computed
                                            with compute_contrasts_maps
    acq_parameters:     List of RT and Flipangle values for ihMT and T1w images
                        [RT, Flipangle]
    Returns
    -------
    ihMT ratio and ihMT saturation matrices in 3D-array.

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
    contrast_maps:      List of 3D-array constrats matrices : list of all
                        contrast maps computed with compute_ihMT_contrasts
    acq_parameters:     Lists of RT and Flipangle for ihMT and T1w
                        [RT, Flipangle]
    Returns
    -------
    MT ratio and MT saturation matrice in 3D-array.
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


def threshold_ihMT_maps(computed_map, contrast_maps, segment_tissue,
                        lower_threshold, upper_threshold, idx_contrast_list):
    """
    Remove NaN and apply different threshold based on
       - maximum and minimum threshold value
       - combination of specific contrasts maps
    Multiple data by sum of T1 probability maps (GM+WM+CSF)

    Parameters
    ----------
    computed_map        3D-Array data.
                        Myelin map (ihMT or non-ihMT maps)
    contrast_maps       List of 3D-Array. File must containing the
                        6 contrasts maps.
    segment_tissue      List of path to tissue probability maps from
                        T1 segmentation
    lower_threshold     Value for low thresold <int>
    upper_thresold      Value for up thresold <int>
    idx_contrast_list   List of indexes of contrast maps corresponding to
                        that of input contrast_maps ex.: [0, 2, 5]
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
    mask = []
    for img in segment_tissue:
        load_image = nib.load(img)
        mask.append(load_image.get_fdata(dtype=np.float64))
    mask_data = mask[0]+mask[1]+mask[2]
    computed_map[np.where(mask_data == 0)] = 0

    # Apply threshold based on combination of specific contrasts maps
    for idx in idx_contrast_list:
        computed_map[contrast_maps[idx] == 0] = 0

    return computed_map


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_output_dirs_exist_and_empty(parser, args,
                                       os.path.join(args.out_dir,
                                                    'Contrats_MT_maps'),
                                       os.path.join(args.out_dir,
                                                    'ihMT_native_maps'),
                                       create_dir=True)

    # Merged all echoes path in a list
    maps = [args.in_altnp, args.in_altpn, args.in_mtoff, args.in_negative,
            args.in_positive, args.in_t1w]

    # Check the number of echoes for each contrast and json file
    for contrast in maps:
        if len(contrast) != args.in_echoes:
            parser.error('The number of images for {} '
                         'must correspond to the number of echoes provide as '
                         'input : {}'.format(os.path.basename(contrast[0]),
                                             args.in_echoes))
        json_file = contrast[0].replace('.nii.gz', '.json')
        if not os.path.isfile(os.path.join(json_file)):
            parser.error('No json file was found for {}'
                         .format(os.path.basename(contrast[0])))

    # Set RT and FlipAngle parameters for ihMT (positive contrast)
    # and T1w images
    parameters = [set_acq_parameters(maps[4][0].replace('.nii.gz',
                                                        '.json')),
                  set_acq_parameters(maps[5][0].replace('.nii.gz',
                                                        '.json'))]

    # Fix issue from the presence of NaN into array
    np.seterr(divide='ignore', invalid='ignore')

    # Define reference image for saving maps
    ref_img = nib.load(maps[4][0])

    # Define contrasts maps names
    contrasts_name = ['altnp', 'altpn', 'reference', 'negative', 'positive',
                      'T1w']
    if args.filtering:
        contrasts_name = [curr_name + '_filter'
                          for curr_name in contrasts_name]

    # Compute contrasts maps
    computed_contrasts = []
    for idx, curr_map in enumerate(maps):
        computed_contrasts.append(compute_contrasts_maps(
                                  curr_map, filtering=args.filtering))

        nib.save(nib.Nifti1Image(computed_contrasts[idx].astype(np.float64),
                                 ref_img.affine, ref_img.header),
                 os.path.join(args.out_dir, 'Contrats_MT_maps',
                              args.id_subj + '_' + contrasts_name[idx]
                              + 'nii.gz'))

    # Compute and thresold ihMT maps
    ihMTR, ihMTsat = compute_ihMT_maps(computed_contrasts, parameters)
    ihMTR = threshold_ihMT_maps(ihMTR, computed_contrasts,
                                args.in_mask, 0, 100,
                                [4, 3, 1, 0, 2])
    ihMTsat = threshold_ihMT_maps(ihMTsat, computed_contrasts,
                                  args.in_mask, 0, 10, [4, 3, 1, 0])

    # Compute and thresold non-ihMT maps
    MTR, MTsat = compute_MT_maps(computed_contrasts, parameters)
    for curr_map in MTR, MTsat:
        curr_map = threshold_ihMT_maps(curr_map, computed_contrasts,
                                       args.in_mask, 0, 100, [4, 2])

    # Save ihMT and MT images as Nifti format
    img_name = ['ihMTR', 'ihMTsat', 'MTR', 'MTsat']
    if args.filtering:
        img_name = [curr_name + '_filter' for curr_name in img_name]

    img_data = ihMTR, ihMTsat, MTR, MTsat
    for img_to_save, name in zip(img_data, img_name):
        nib.save(nib.Nifti1Image(img_to_save.astype(np.float64),
                                 ref_img.affine, ref_img.header),
                 os.path.join(args.out_dir, 'ihMT_native_maps',
                              args.id_subj + '_' + name + '.nii.gz'))


if __name__ == '__main__':
    main()
