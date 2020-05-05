# -*- coding: utf-8 -*-
"""
Computing ihMT and non-ihMT maps
Make a more details description !
"""
# Verifier si import dans le bon ordre...
import argparse
from bids import BIDSLayout
import math

import numpy as np
import nibabel as nib
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
                   help="ihMT sequences resolution as set in BIDSFile object:"
                         "ex. 'highres' or 'lowres'. Default is None.")
    p.add_argument('--filtering', action='store_true',
                   help='Gaussian filtering to remove Gibbs ringing,'
                        'sigma set at 0.5. Not recommanded.'
                        'Default is None')

    add_overwrite_arg(p)

    return p


def set_acq_parameters(image):
    TR = (layout.get_metadata(image)['RepetitionTime'])*1000
    FlipAngle = ((layout.get_metadata(image)['FlipAngle'])*math.pi/180)
    return TR, FlipAngle


def merge_ihMT_array(images):
    """
    Function to concatenate n 3D-array matrix of the same size
    along the 4th dimension.

    Parameters
    ----------
    images       list of echo path. Must be nii image.

    Returns
    -------
    Return a matrix (x,y,z,n) for n matrices (x,y,z) as input
    """
    merge_array = []
    for echo in range(len(images)):
        nii_image = nib.load(images[echo])
        merge_array.append(np.array(nii_image.dataobj))
    merge_array = np.stack(merge_array, axis=-1)

    # @Arnaud : p-e un moyen  plus simple de le faire ?
    orig_shape = str(np.array(nii_image.dataobj).shape).replace(')', ', ' +
                     str(len(images)) + ')')

    # @Arnaud : je ne suis pas certaine de la fct à utiliser pour
    # stopper en cas d'erreur : exit ? break ?
    # j'ai vue qu'il y avait parser.error mais fctne que pour parser non ?

    # Assert the goodness of fit of 4D merged file
    if (str(merge_array.shape) == orig_shape) == False:
        print('Error: the number of 4th dimension does '
                     'not correspond to the number of echoes')

    return merge_array


def py_fspecial_gauss(shape, sigma):
    """
    Function to mimic the 'fspecial gaussian' MATLAB function
    Returns a rotationally symmetric Gaussian lowpass filter of
    shape size with standard deviation sigma.
    see https://www.mathworks.com/help/images/ref/fspecial.html

    Parameters
    ----------
    shape    vector to specify the size of square matrix (rows, columns).
             ex.: (3, 3)
    sigma    Standard deviation (0.5 is recommanded).

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


# @Arnaud : comment faire en sorte que le resolution soit None/opt
#           pour resolution et filtering ?
def compute_ihMT_contrast(contrast_maps, ref_image, output_path, output_name,
                          resolution=None, filtering=None):
    """
    Compute contrasts maps : more description details !

    Parameters
    ----------
    contrast_maps   List of BIDSFile object : list of echos image path
    ref_image       Reference volume to set header
    resolution      Resolution of ihMT sequences as set in BIDSFile object:
                    ex. 'highres' or 'lowres', (default: None).
    filtering       Apply Gaussian filtering to remove Gibbs ringing
                    (default: None).
    Returns
    -------
    Contrast Map in NxN array and save contrast Map in Nifti format.
    """

    # Merged different echo image into 4D-array
    merged_map = merge_ihMT_array(contrast_maps)

    # Compute the sum of contrast map
    contrast_map = np.sqrt(np.sum(np.squeeze(merged_map**2), 3))

    # Apply gaussian filtering if needed
    if filtering is not None:
        h = py_fspecial_gauss((3, 3), 0.5)
        contrast_map = scipy.ndimage.convolve(contrasts, h).astype(float)

    # Save contrast maps in nii.gz format
    if resolution is not None:
        save_ihMT_maps(ref_image, contrast_map, output_path,
                       output_name, resolution)
    else:
        save_ihMT_maps(ref_image, contrast_map, output_path,
                       output_name, resolution=None)

    return contrast_map


def compute_ihMT_map(contrast_maps, acq_parameters):
    """
    Compute ihMT maps : more description details !

    Parameters
    ----------
    contrast_maps:      List of BIDSFile object : list of all contrast maps
    acq_parameters:     Lists of parameters for ihMT and T1w
                        [TR and Flipangle]
    Returns
    -------
    ihMT ratio and saturation Maps in Nifti format.
    """
    # Compute ihMTratio map
    ihMTR = 100*(computed_maps[4]+computed_maps[3]-computed_maps[1]
                 - computed_maps[0])/computed_maps[2]

# flake8 to fix: longueur ligne ... jamais content !
    # Compute ihMTsat map
    # Compute an dR1sat image (Varma et al. ISMRM 2015)
    cPD1a = (computed_maps[4]+computed_maps[3])/2
    cPD1b = (computed_maps[1]-computed_maps[0])/2
    cT1 = computed_maps[5]
    T1appa = ((cPD1a/acq_parameters[0][1])-(cT1/acq_parameters[1][1]))/((cT1*acq_parameters[1][1]) /
             (2*acq_parameters[1][0]/1000)-(cPD1a*acq_parameters[0][1])/(2*acq_parameters[0][0]/1000))
    T1appb = ((cPD1b/acq_parameters[0][1])-(cT1/acq_parameters[1][1]))/((cT1*acq_parameters[1][1]) /
             (2*acq_parameters[1][0]/1000)-(cPD1b*acq_parameters[0][1])/(2*acq_parameters[0][0]/1000))
    ihMTsat = (1/T1appb)-(1/T1appa)

    return ihMTR, ihMTsat


def compute_MT_map(computed_maps, acq_parameters):
    """
    Compute MT maps : more description details !

    Parameters
    ----------
    contrast_maps:      List of BIDSFile object : list of all contrast maps
    acq_parameters:     Lists of parameters for ihMT and T1w
                        [TR and Flipangle]
    Returns
    -------
    Non-ihMT maps : MT ratio and saturation Maps in Nifti format.
    """
    # Compute MTR map
    MTR = ((computed_maps[2]-(computed_maps[4]+computed_maps[3])/2) /
           computed_maps[2])*100

# flake8 to fix: longueur ligne ... jamais content !
    # Compute MTsat maps
    cPD1 = computed_maps[2]
    cPD2 = (computed_maps[4]+computed_maps[3])/2
    cT1 = computed_maps[5]
    Aapp = ((2*acq_parameters[0][0]/(acq_parameters[0][1]**2))-(2*acq_parameters[1][0] /
            (acq_parameters[1][1]**2)))/(((2*acq_parameters[0][0]) /
            (acq_parameters[0][1]*cPD1))-((2*acq_parameters[1][0])/(acq_parameters[1][1]*cT1)))
    T1app = ((cPD1/acq_parameters[0][1]) - (cT1/acq_parameters[1][1]))/((cT1*acq_parameters[1][1]) /
             (2*acq_parameters[1][0])-(cPD1*acq_parameters[0][1])/(2*acq_parameters[0][0]))
    MTsat = ((Aapp*acq_parameters[0][1]*acq_parameters[0][0]/T1app)/cPD2)-(acq_parameters[0][0]/T1app)-
            (acq_parameters[0][1]**2)/2

    return MTR, MTsat


def threshold_ihMT_maps(computed_map, contrast_maps, brainmask,
                        lower_threshold, upper_threshold, contrast_maps_list):
    """
    Remove NaN and Apply different threshold based on
       - max and min threshold value
       - brainmask
       - combination of specific contrasts maps

    Parameters
    ----------
    computed_map        ihmt or non-ihmt computed maps
    brainmask           Path to T1 binary brainmask
    lower_threshold     Value for low thresold <int>
    upper_thresold      Value for up thresold <int>
    contrast_maps_list  List of contrasts maps
                        ex.: ['positive', 'negative', 'reference']
    Returns
    ----------
    Thresholded map  NxN array
    """

    # Remove NaN and apply thresold based on lower and upper value
    computed_map[np.isnan(computed_map)] = 0
    computed_map[np.isinf(computed_map)] = 0
    computed_map[computed_map < lower_threshold] = 0
    computed_map[computed_map > upper_threshold] = 0

    # Load and apply brainmasks on maps
    mask_image = nib.load(brainmask)
    mask = np.array(mask_image.dataobj)
    computed_map = computed_map*mask

    # Apply threshold based on combination of specific contrasts maps
    for idx in contrast_maps_list:
        computed_map[contrast_maps[int(idx)] == 0] = 0

    return computed_map


# @Arnaud : toujours cette question sur les opt en function
# adapter le resolution == 1
def save_ihMT_maps(ref_image, image_to_save, output_path, output_name,
                   resolution, filtering=None):
    """ Save matrix into nii images

    Parameters
    ----------
    ref_image       Reference volume to set image parameter
    image_to_save   Matrix array to save in Nifti image.
    output_path     Path to save image
    output_name     Image filename
    resoltuion      Resolution of ihMT sequences as set in BIDSFile object:
                    ex. 'highres' or 'lowres', (default: None).
    """
    ref_img = nib.load(ref_image)
    if resolution:
        nib.save(nib.Nifti1Image(image_to_save.astype(np.float64),
                                 ref_img.affine, ref_img.header),
                 os.path.join(output_path, output_name + '_' +
                              resolution + '.nii'))

    else:
        nib.save(nib.Nifti1Image(image_to_save.astype(np.float64),
                                 ref_img.affine, ref_img.header),
                 os.path.join(output_path, output_name + '.nii'))

#    if filetring ...?


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Verify existence of input and output
    assert_inputs_exist(parser, [args.input,
                                 args.brainmask,
                                 args.subjid])
    assert_outputs_exist(parser, [args.output])

# a voir pour utiliser glob.glob pour créer structure fichier
    # Select contrasts files
    maps = []
    layout = BIDSLayout(args.input, validate=False)
    acquisition = layout.get_acquisition()

    if args.resolution:
        acquisition = acquisition[0::len(args.resolution)]

# !!! attention resoudre le problème du if
    for acq in acquisition:
        maps.append(layout.get(subject=args.subjid,
                                   datatype='anat', suffix='ihmt',
                                   aquisition=(acq.replace(arg.resolution[0], ''))
                                                           + args.resolution,
                                   extension='nii.gz', return_type='file'))
        else:
            maps.append(layout.get(subject=args.subjid, datatype='anat',
                                   suffix='ihmt', aquisition=acq,
                                   extension='nii.gz', return_type='file'))

    # Set TR and FlipAngle parameters for ihmt and t1w images
    parameters = []
    parameters.append(set_acq_parameters(maps[4][0]))
    parameters.append(set_acq_parameters(maps[5][0]))

    # Fix issue from the presence of NaN into array
    np.seterr(divide='ignore', invalid='ignore')

    # Compute contrasts maps   !! a revoir avec ces fichus conditions !!
    computed_maps = []
    outname = ['altnp', 'altpn', 'reference', 'negative', 'positive', 'T1w']
    for idx, curr_map in enumerate(maps):
        if args.resolution and args.filtering:
            computed_maps.append(compute_ihMT_contrast(curr_map, maps[4][0],
                                                       args.output,
                                                       outname[idx],
                                                       args.resolution,
                                                       filtering))

        if args.resolution:
            computed_maps.append(compute_ihMT_contrast(curr_map, maps[4][0],
                                                       args.output,
                                                       outname[idx],
                                                       args.resolution))
        if args.filtering:
            computed_maps.append(compute_ihMT_contrast(curr_map, maps[4][0],
                                                       args.output,
                                                       outname[idx],
                                                       filtering))


    # Compute ihMT maps
    # Compue ihMTratio map
    ihMTR, ihMTsat = compute_ihMT_map(computed_maps, parameters)
    ihMTR_list = ['positive', 'negative', 'altPN', 'reference']
    ihMTR = threshold_ihMT_maps(ihMTR, arg.brainmask, 0, 100,
                                ihMTR_list)
    save_ihMT_maps(maps[4][0], ihMTR, 'ihMTR', args.resolution)

    # Compue ihMTsat map
    ihMTsat_list = ['positive', 'negative', 'altPN', 'altNP']
    ihMTsat = threshold_ihMT_maps(ihMTsat, arg.brainmask, 0, 10,
                                  ihMTsat_list)
    save_ihMT_maps(maps[4][0], ihMTsat, 'ihMT_dR1sat',
                   args.resolution)


    # Compute non-ihMT maps
    MT_maps = []
    MT_maps.append(compute_MTR(computed_maps, parameters))
    MT_list = ['positive', 'reference']
    for curr_map in MT_maps:
        curr_map = threshold_ihMT_maps(curr_map, arg.brainmask, 0, 100,
                                       MT_list)
        # save images
        save_ihMT_maps(maps[4][0], MT_maps[0][0], 'MTR',
                       args.resolution)
        save_ihMT_maps(maps[4][0], MT_maps[0][1], 'MTsat',
                       args.resolution)


if __name__ == '__main__':
    main()
