# -*- coding: utf-8 -*-
"""
Computing ihMT and non-ihMT maps
Make a more details description !
"""

import argparse
import os

import ants
from bids import BIDSLayout
import nibabel as nib
import numpy as np
import scipy.ndimage

from scilpy.io.utils import (add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_bids_folder",
                   help='Path of the ihMT maps from bids folder')
    p.add_argument('out_dir',
                   help='Path to output folder')
    p.add_argument('in_participant',
                   help='Id of subject from BIDS folder. Ex : "01"')
    p.add_argument('in_mask',
                   help='Path to T1 probability brainmask.'
                        'Could the sum of 3 (WM+GM+CSF) brain compartments.')
    p.add_argument('--resolution', default=None, nargs='+',
                   help="Name of ihMT sequences resolution. Use this option"
                        "when 2 or most resolution are acquired."
                        "Name of ihMT sequences resolution must be as set in"
                        "BIDSFile object. Ex:'highres' "
                        "Default is None.")
    p.add_argument('--filtering', default=None,
                   help='Gaussian filtering to remove Gibbs ringing,'
                        'sigma set at 0.5. Not recommanded.'
                        'Default is None')

    add_overwrite_arg(p)

    return p


def prepare_images(in_maps, fixed_img, out_dir, out_trf_name):
    """
    Function to register each echo of images to a reference image
    using the python version of  ANTs registration.
    https://antspy.readthedocs.io/en/latest/registration.html

    Parameters
    ----------
    in_maps          List of Bids file object
    fixed_img        Path to reference image to which we register
                     the moving image.
    out_dir          Path to output directory to save warped images
    out_trf_name     Name of Transforms output

    Returns
    ----------
    Return a list of 3D-array matrix of images warped
    """

    warped_img_array = []
    ref_img = ants.image_read(fixed_img)
    source_img = ants.image_read(in_maps[0])

    # Compute coresgister transformation
    coreg_mat = ants.registration(fixed=ref_img, moving=source_img,
                                  type_of_transform='Rigid',
                                  outprefix=os.path.join(out_dir,
                                  'Coregistration_maps', (out_trf_name)))

    # Apply coresgister transformation on echos
    for echo in range(len(in_maps)):
        img_to_coreg = ants.image_read(in_maps[echo])
        out_name = os.path.basename(in_maps[echo]).replace('.nii.gz', '')
        coreg_img = ants.apply_transforms(fixed=ref_img,
                                          moving=img_to_coreg,
                                          transformlist=coreg_mat['fwdtransforms'])
        warped_img_array.append(coreg_img.numpy())
        ants.image_write(coreg_img, os.path.join(out_dir,
                                                 'Coregistration_maps',
                                                 out_name + '_warped.nii.gz'))

    return warped_img_array


def set_acq_parameters(bidsLayout, image_path):
    TR = (bidsLayout.get_metadata(image_path)['RepetitionTime'])*1000
    FlipAngle = ((bidsLayout.get_metadata(image_path)['FlipAngle'])*np.pi/180)
    return TR, FlipAngle


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
    sigma    Value for standard deviation (0.5 is recommanded).

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


def compute_contrasts_maps(echo_images, filtering=None):
    """
    Load echo images and compute contrasts maps : more description details !

    Parameters
    ----------
    echo_images     List of BIDSFile object : list of echo(s) path for each
                    contrasts in Bids folder
    filtering       Apply Gaussian filtering to remove Gibbs ringing
                    (default: None).

    Returns
    -------
    Returns contrast map in 3D-Array.
    """

    # Merged the 3 echo images into 4D-array
    # merged_map = merge_ihMT_array(echo_images)
    merged_map = np.stack(echo_images, axis=-1)

    # Compute the sum of contrast map
    contrast_map = np.sqrt(np.sum(np.squeeze(merged_map**2), 3))

    # Apply gaussian filtering if needed
    if filtering:
        h = py_fspecial_gauss((3, 3), 0.5)
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
    acq_parameters:     Lists of parameters for ihMT and T1w
                        [TR and Flipangle]
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
    contrast_maps:      List of constrats : list of all contrast maps computed
                                            with compute_ihMT_contrasts
    acq_parameters:     Lists of parameters for ihMT and T1w
                        [TR and Flipangle]
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
    Remove NaN and Apply different threshold based on
       - max and min threshold value
       - brainmask
       - combination of specific contrasts maps

    Parameters
    ----------
    computed_map        3D-Array data.
                        Myelin map (ihMT or non-ihMT computed maps)
    contrast_maps       List of 3D-Array. File must containing the
                        5 contrasts maps.
    in_mask             Path to binary brainmask
    lower_threshold     Value for low thresold <int>
    upper_thresold      Value for up thresold <int>
    idx_contrast_list   List of indice contrasts maps in order corresponding to
                        contrast_maps input ex.: [1, 2, 3]
                        Altnp = 0; Atlpn = 1; Reference = 2; Negative = 3;
                        Positive = 4; T1weighted = 5
    Returns
    ----------
    Thresholded map in 3D-array.
    """
    # Remove NaN and apply thresold based on lower and upper value
    computed_map[np.isnan(computed_map)] = 0
    computed_map[np.isinf(computed_map)] = 0
    computed_map[computed_map < lower_threshold] = 0
    computed_map[computed_map > upper_threshold] = 0

    # Load and apply brainmasks on maps
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

    # Select contrasts files from Bids folder
    maps = []
    layout = BIDSLayout(args.in_bids_folder, validate=False)
    acquisition = layout.get_acquisition()

    if args.resolution:
        curr_res_acq = [res for res in acquisition if args.resolution[0] in res]
        for idx, curr_acq in enumerate(curr_res_acq):
            maps.append(layout.get(subject=args.in_participant,
                                   datatype='anat', suffix='ihmt',
                                   acquisition=str(curr_acq),
                                   extension='nii.gz', return_type='file'))
    else:
        for curr_acq in acquisition:
            maps.append(layout.get(subject=args.in_participant, datatype='anat',
                                   suffix='ihmt', acquisition=str(curr_acq),
                                   extension='nii.gz', return_type='file'))

    # Set TR and FlipAngle parameters for ihMT and T1w images
    parameters = []
    parameters.append(set_acq_parameters(layout, maps[4][0]))
    parameters.append(set_acq_parameters(layout, maps[5][0]))

    # Fix issue from the presence of NaN into array
    np.seterr(divide='ignore', invalid='ignore')

    # Define reference image for saving ihMT files
    ref_img = nib.load(maps[4][0])

    # Define image output Name
    if args.resolution:
        out = [res for res in acquisition if args.resolution[0] in res]
    else:
        out = acquisition

    # Create Coregistration images folers
    if not os.path.isdir(os.path.join(args.out_dir, 'Coregistration_maps')):
        os.mkdir(os.path.join(args.out_dir, 'Coregistration_maps'))

    # Compute coregistration maps
    warped_img = []
    for idx, curr_acq in enumerate(maps):
        if args.resolution:
            warped_img.append(prepare_images(maps[idx], maps[4][0], args.out_dir,
                                             out[idx].replace(args.resolution[0], '')))
        else:
            warped_img.append(prepare_images(maps[idx], maps[4][0], args.out_dir,
                                             out[idx].replace(args.resolution[0], '')))

    # Create contrasts maps folers
    if not os.path.isdir(os.path.join(args.out_dir, 'Contrats_maps')):
        os.mkdir(os.path.join(args.out_dir, 'Contrats_maps'))

    # Compute contrasts maps
    computed_contrasts = []
    for idx, curr_map in enumerate(warped_img):
        if args.resolution and args.filtering:
            computed_contrasts.append(compute_contrasts_maps(curr_map,
                                                             filtering=True))
            nib.save(nib.Nifti1Image(computed_contrasts[idx].astype(np.float64),
                                     ref_img.affine, ref_img.header),
                     os.path.join(args.out_dir, 'Contrats_maps',
                                  out[idx].replace(args.resolution[0], '') +
                                  '_' + args.resolution[0] + '_filter.nii'))

        elif args.resolution:
            computed_contrasts.append(compute_contrasts_maps(curr_map))
            nib.save(nib.Nifti1Image(computed_contrasts[idx].astype(np.float64),
                                     ref_img.affine, ref_img.header),
                     os.path.join(args.out_dir, 'Contrats_maps',
                                  out[idx].replace(args.resolution[0], '')
                                  + '_' + args.resolution[0] + '.nii'))

        elif args.filtering:
            computed_contrasts.append(compute_contrasts_maps(curr_map,
                                                             filtering=True))
            nib.save(nib.Nifti1Image(computed_contrasts[idx].astype(np.float64),
                                     ref_img.affine, ref_img.header),
                     os.path.join(args.out_dir, 'Contrats_maps',
                                  out[idx] + '_filter.nii'))
        else:
            computed_contrasts.append(compute_contrasts_maps(curr_map))
            nib.save(nib.Nifti1Image(computed_contrasts[idx].astype(np.float64),
                                     ref_img.affine, ref_img.header),
                     os.path.join(args.out_dir, 'Contrats_maps',
                                  out[idx] + '.nii'))

    # Create ihMT and non-ihMT maps folers
    if not os.path.isdir(os.path.join(args.out_dir, 'ihMT_native_maps')):
        os.mkdir(os.path.join(args.out_dir, 'ihMT_native_maps'))

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
    img_data = ihMTR, ihMTsat, MTR, MTsat
    img_name = ['ihMTR', 'ihMTsat', 'MTR', 'MTsat']

    for img_to_save, name in zip(img_data, img_name):
        if args.resolution and args.filtering:
            nib.save(nib.Nifti1Image(img_to_save.astype(np.float64),
                                     ref_img.affine, ref_img.header),
                     os.path.join(args.out_dir, 'ihMT_native_maps', name + '_'
                                  + args.resolution[0] + '_filter.nii'))

        elif args.resolution:
            nib.save(nib.Nifti1Image(img_to_save.astype(np.float64),
                                     ref_img.affine, ref_img.header),
                     os.path.join(args.out_dir, 'ihMT_native_maps', name + '_'
                                  + args.resolution[0] + '.nii'))

        elif args.filtering:
            nib.save(nib.Nifti1Image(img_to_save.astype(np.float64),
                                     ref_img.affine, ref_img.header),
                     os.path.join(args.out_dir, 'ihMT_native_maps', name +
                                  '_filter.nii'))
        else:
            nib.save(nib.Nifti1Image(img_to_save.astype(np.float64),
                                     ref_img.affine, ref_img.header),
                     os.path.join(args.out_dir, 'ihMT_native_maps', name
                                  + '.nii'))


if __name__ == '__main__':
    main()
