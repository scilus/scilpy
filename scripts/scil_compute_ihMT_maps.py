# -*- coding: utf-8 -*-
"""
Computing ihMT and non-ihMT maps
Make a more details description !
"""
# Verifier si import dans le bon ordre...
import argparse
from bids import BIDSLayout
import json
import os
import math

import numpy as np
import nibabel as nib
from pathlib import Path
import scipy.ndimage

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=args.RawTextHelpFormatter)
    p.add_argument('input', metavar='input',
                   help='Path of the ihMT maps from bids folder')
    p.add_argument('output', metavar='output',
                   help='Path to output folder')
    p.add_argument('subjid', metavar='subjid',
                   help='Id of subject from BIDS folder')
    p.add_argument('brainmask', metavar='input',
                   help='Path to T1 binary brainmask.'
                        'Generate by BET or other tools')
    p.add_argument('--resolution', action='store_true', metavar='string',
                   help="Give resolution ihmt: 'highres' or 'lowres'")
    p.add_argument('--filtering', action='store_true',
                   help='Gaussian filtering to remove Gibbs ringing,'
                        'sigma set at 0.5. Not recommanded.'
                        'Default is None')

    add_overwrite_arg(p)

    return p


def set_parameters(image):
    TR = (layout.get_metadata(image)['RepetitionTime'])*1000
    FlipAngle = ((layout.get_metadata(image)['FlipAngle'])*math.pi/180)
    return TR, FlipAngle


def merge_ihMT_array(image, nb_echoes):
    """
    Function to concatenate n 3D-array matrix of the same size
    along the 4th dimension.
    Return a matrix (x,y,z,n) for n matrices (x,y,z) as input
    """
    merge_array = []
    for k in range(0, nb_echoes):
        nii_image = nib.load(image[k])
        nii_array = np.array(nii_image.dataobj)
        merge_array.append(nii_array)
        # merge_array.append(np.array(nii_image.dataobj))
    merge_array = np.stack(merge_array, axis=-1)
    return merge_array


def py_fspecial_gauss(shape, sigma):
    """
    Function to mimic the 'fspecial gaussian' MATLAB function
    Returns a rotationally symmetric Gaussian lowpass filter of
    shape size with standard deviation sigma.
    see https://www.mathworks.com/help/images/ref/fspecial.html
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def compute_ihMT_contrast(contrast_maps, ref_image, acq_resolution=None,
                          filtering=None):
    """
    Compute contrasts maps in Nifti and array format

    Parameters
    ----------
    contrast_map         List of BIDSFile object : list of contrast maps
    ref_image            Give reference images to set header
    resolution           Give ihMT resolution.
    filtering            Apply Gaussian filtering. Default is None.

    Returns
    -------
    Contrast Map in NxN array
    + Save contrast Map in Nifti format in current folder
    """
    resulting_map = []
    NbEchoes = len(contrast_maps[0])

    # Merged different echo into 4d-array
    merged_map = merge_ihMT_array(contrast_maps, NbEchoes)

    # Compute the contrast map
    contrast_map = np.sqrt(np.sum(np.squeeze(merged_map**2), 3))

    # Apply gaussian filtering if needed, mainly to remove Gibbs ringing
    if args.filtering is not None:
        h = py_fspecial_gauss((3, 3), 0.5)
        contrast_map = scipy.ndimage.convolve(contrasts, h).astype(float)

    resulting_map.append(contrast_map)

    # Save contrast maps in nii.gz format
    if args.acq_resolution is not None:
        save_maps(ref_image, contrast_map, contrast_map_name, acq_resolution)
    else:
        save_maps(ref_image, contrast_map, contrast_map_name)

    return resulting_map


def compute_ihMT_map(contrast_maps):
    """ description
    """
    # Compute ihMTratio map
    ihMTR = 100*(computed_maps[4]+computed_maps[3]-computed_maps[1]
                 - computed_maps[0])/computed_maps[2]

    # Compute ihMTsat map
    # Compute an dR1sat image (Varma et al. ISMRM 2015)
    cPD1a = (computed_maps[4]+computed_maps[3])/2
    cPD1b = (computed_maps[1]-computed_maps[0])/2
    cT1 = computed_maps[5]
    T1appa = ((cPD1a/FlipAngleMT)-(cT1/FlipAngleT1))/((cT1*FlipAngleT1) /
             (2*TR_T1/1000)-(cPD1a*FlipAngleMT)/(2*TR_MT/1000))
    T1appb = ((cPD1b/FlipAngleMT)-(cT1/FlipAngleT1))/((cT1*FlipAngleT1) /
             (2*TR_T1/1000)-(cPD1b*FlipAngleMT)/(2*TR_MT/1000))
    ihMTsat = (1/T1appb)-(1/T1appa)

    return ihMTRatio, ihMTsat


def compute_MT_map(computed_maps):
    """ description
    """
    # Compute MTR map
    MTR = ((computed_maps[2]-(computed_maps[4]+computed_maps[3])/2) /
           computed_maps[2])*100

    # Compute MTsat maps
    cPD1 = computed_maps[2]
    cPD2 = (computed_maps[4]+computed_maps[3])/2
    cT1 = computed_maps[5]
    Aapp = ((2*TR_MT/(FlipAngleMT*FlipAngleMT))-(2*TR_T1 /
            (FlipAngleT1*FlipAngleT1)))/(((2*TR_MT) /
            (FlipAngleMT*cPD1))-((2*TR_T1)/(FlipAngleT1*cT1)))
    T1app = ((cPD1/FlipAngleMT) - (cT1/FlipAngleT1))/((cT1*FlipAngleT1) /
             (2*TR_T1)-(cPD1*FlipAngleMT)/(2*TR_MT))
    MTsat = ((Aapp*FlipAngleMT*TR_MT/T1app)/cPD2)-(TR_MT/T1app)-
            (FlipAngleMT*FlipAngleMT)/2

    return MTR, MTsat


def threshold_ihMT_maps(computed_map, brainmask, lower_threshold,
                        upper_thresold, contrast_maps_list):
    """
    Remove NaN and Apply different threshold based on
       - max and min threshold value
       - brainmask
       - combination of specific contrasts maps

    Parameters
    ----------
    computed_map        Name of ihmt computed maps
    brainmask           Path to T1 binary brainmask
    lower_threshold     Value for low thresold <int>
    upper_thresold      Value for up thresold <int>
    contrast_maps_list  List of contrasts maps
                        example : ['Positive', 'Negative']
    Returns
    ----------
    Maps thresholded  NxN array
    """

    # Remove NaN and apply thresold based on lower and upper value
    computed_map[isnan(computed_map)] = 0
    computed_map[isinf(computed_map)] = 0
    computed_map[computed_map < lower_threshold] = 0
    computed_map[computed_map > upper_threshold] = 0

    # Load and apply brainmasks on maps
    mask_image = nib.load(brainmask)
    mask = np.array(mask_image.dataobj)
    computed_map = computed_map*mask

    # Apply combination of specific contrasts maps threshold
    for contrast_maps in contrast_maps_list:
        computed_map[contrast_maps == 0] = 0

    return computed_map


def save_ihMT_maps(ref_image, data, output_name, resoltuion):
    """ description (modifier pour plus simple cf ref_image.affine)
    + demander si c'est ok avec resolution if.arg..."""
    ref_img = nib.load(ref_imgage)
    affine = ref_img.get_affine()
    hdr = nib.Nifti1Header()
    hdr.set_slope_inter(1.0, 0.0)
    hdr['glmax'] = data.max()
    hdr['glmin'] = data.min()
    img = nib.Nifti1Image(data.astype(np.float64), affine, hdr)
    if arg.resolution is not None:
        nib.save(img, os.path.join(args.ouput,
                                   output_name + '_' + resolution + '.nii'))
    else:
        nib.save(img, os.path.join(args.ouput,
                                   output_name + '.nii'))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Verify existence of input and output
    assert_inputs_exist(parser, [args.input,
                                 args.brainmask,
                                 args.subjid])
    assert_outputs_exist(parser, [args.output])

    # Select Contrasts files utiliser glob.glob pour créer structure fichier
    maps = []
    layout = BIDSLayout(args.input, validate=False)
    acquisition = layout.get_acquisition()
    for acq in acquisition:
        if args.resolution:
            res = acq[0::2]
            maps.append(layout.get(subject=args.subjid,
                                   datatype='anat', suffix='ihmt',
                                   aquisition=(res.replace('highres', ''))
                                                           + args.resolution,
                                   extension='nii.gz', return_type='file'))
        else:
            maps.append(layout.get(subject=args.subjid, datatype='anat',
                                   suffix='ihmt', aquisition=acq,
                                   extension='nii.gz', return_type='file'))

    # Set TR and FlipAngle parameters
    TR_MT, FlipAngleMT = set_parameters(maps[4][0])
    TR_T1w, FlipAngleT1w = set_parameters(maps[4][0])

    # Fix issue from the presence of NaN into array
    np.seterr(divide='ignore', invalid='ignore')

    # Compute contrasts maps
    computed_maps = []
    for curr_map in maps:
        if args.filtering:
            computed_maps.append(compute_contrast(cur_map, maps[4][0],
                                                  filtering))
        else:
            computed_maps.append(compute_contrast(cur_map, maps[4][0]))

    # Compute ihMT maps
    # Compue ihMTratio map
    ihMTR, ihMTsat = compute_ihMT_map(computed_maps)
    ihMTR_list = ['Positive', 'Negative', 'AltPN', 'reference']
    ihMTR = threshold_ihMT_maps(ihMTR, arg.brainmask, 0, 100,
                                ihMTR_list)
    save_ihMT_maps(computed_maps[4][0], ihMTR, 'ihMTR', args.resolution)

    # Compue ihMTsat map
    ihMTsat_list = ['positive', 'negative', 'altPN', 'altNP']
    ihMTsat = threshold_ihMT_maps(ihMTsat, arg.brainmask, 0, 10,
                                  ihMTsat_list)
    save_ihMT_maps(computed_maps[4][0], ihMTsat, 'ihMT_dR1sat',
                   args.resolution)

    # Compute non-ihMT maps
    MT_maps = []
    MT_maps.append(compute_MTR(computed_maps))
    MT_list = ['Positive', 'reference']
    for curr_map in MT_maps:
        curr_map = threshold_ihMT_maps(curr_map, arg.brainmask, 0, 100,
                                       MT_list)
        # save image
        save_ihMT_maps(computed_maps[4][0], MT_maps[0], 'MTR',
                       args.resolution)
        save_ihMT_maps(computed_maps[4][0], MT_maps[1], 'MTsat',
                       args.resolution)


if __name__ == '__main__':
    main()
