#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script computes two myelin indices maps from the Magnetization Transfer
(MT) images.
Magnetization Transfer is a contrast mechanism in tissue resulting from the
proton exchange between non-aqueous protons (from macromolecules and their
closely associated water molecules, the "bound" pool) and protons in the free
water pool called aqueous protons. This exchange attenuates the MRI signal,
introducing microstructure-dependent contrast. MT's effect reflects the
relative density of macromolecules such as proteins and lipids, it has been
associated with myelin content in white matter of the brain.

Different contrasts can be done with an off-resonance pulse to saturating the
protons on non-aqueous molecules a frequency irradiation. The MT maps are
obtained using three contrasts: single frequency irradiation (MT-on, saturated
images) and an unsaturated contrast (MT-off); and a T1weighted image as
reference.

The output consist in two types of images:
        Three contrasts images : MT-off, MT-on and T1weighted images.
        MT maps corrected or not for an empiric B1 correction maps.

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
  1. Contrasts_MT_maps which contains the 2 contrast images
      - MT-off.nii.gz : pulses applied at positive frequency
      - MT-on.nii.gz : pulses applied at negative frequency
      - T1w.nii.gz : anatomical T1 reference images


  2. MT_native_maps which contains the 4 myelin maps
      - MTR.nii.gz : Magnetization Transfer Ratio map
      The MT ratio is a measure reflecting the amount of bound protons.

      - MTsat.nii.gz : Magnetization Transfer saturation map
      The MT saturation is a pseudo-quantitative maps representing
      the signal change between the bound and free water pools.

>>> scil_mti_maps_MT.py path/to/output/directory path/to/mask_bin.nii.gz
    --in_mtoff path/to/echo*mtoff.nii.gz --in_mton path/to/echo*mton.nii.gz
    --in_t1w path/to/echo*T1w.nii.gz

Formerly: scil_compute_MT_maps.py
"""

import argparse
import os

import nibabel as nib
import numpy as np

from scilpy.io.utils import (get_acq_parameters, add_overwrite_arg,
                             assert_inputs_exist, add_verbose_arg,
                             assert_output_dirs_exist_and_empty)
from scilpy.io.image import load_img
from scilpy.image.volume_math import concatenate
from scilpy.reconst.mti import (compute_contrasts_maps,
                                compute_MT_maps, threshold_maps,
                                apply_B1_correction)

EPILOG = """
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

    g = p.add_argument_group(title='MT contrasts', description='Path to '
                             'echoes corresponding to contrasts images. All '
                             'constrasts must have the same number of echoes '
                             'and coregistered between them.'
                             'Use * to include all echoes.')
    g.add_argument("--in_mtoff", nargs='+',
                   help='Path to all echoes corresponding to the '
                        'no frequency saturation pulse (reference image).')
    g.add_argument("--in_mton", nargs='+',
                   help='Path to all echoes corresponding to the '
                        'Positive frequency saturation pulse.')
    g.add_argument("--in_t1w", nargs='+',
                   help='Path to all echoes corresponding to the '
                        'T1-weigthed.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_output_dirs_exist_and_empty(parser, args,
                                       os.path.join(args.out_dir,
                                                    'Contrasts_MT_maps'),
                                       os.path.join(args.out_dir,
                                                    'MT_native_maps'),
                                       create_dir=True)

    # Merge all echos path into a list
    maps = [args.in_mtoff, args.in_mton, args.in_t1w]

    maps_flat = (args.in_mtoff + args.in_mton + args.in_t1w)

    jsons = [curr_map.replace('.nii.gz', '.json')
             for curr_map in maps_flat]

    # check data
    assert_inputs_exist(parser, jsons + maps_flat)
    for curr_map in maps[1:]:
        if len(curr_map) != len(maps[0]):
            parser.error('Not the same number of echoes per contrast')

    # Set TR and FlipAngle parameters for MT (mtoff contrast)
    # and T1w images
    parameters = []
    for curr_map in maps[0][0], maps[2][0]:
        acq_parameter = get_acq_parameters(curr_map.replace('.nii.gz',
                                                            '.json'),
                                           ['RepetitionTime', 'FlipAngle'])
        acq_parameter = acq_parameter[0]*1000, acq_parameter[1]*np.pi/180
        parameters.append(acq_parameter)

    # Fix issue from the presence of invalide value and division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Define reference image for saving maps
    ref_img = nib.load(maps[0][0])

    # Define contrasts maps names
    contrasts_name = ['mt_off', 'mt_on', 'T1w']

    if args.out_prefix:
        contrasts_name = [args.out_prefix + '_' + curr_name
                          for curr_name in contrasts_name]

    # Compute contrasts maps
    computed_contrasts = []
    for idx, curr_map in enumerate(maps):
        input_images = []
        for image in curr_map:
            img, _ = load_img(image)
            input_images.append(img)
        merged_curr_map = concatenate(input_images, input_images[0])

        computed_contrasts.append(compute_contrasts_maps(
            merged_curr_map, filtering=args.filtering))

        nib.save(nib.Nifti1Image(computed_contrasts[idx].astype(np.float32),
                                 ref_img.affine),
                 os.path.join(args.out_dir, 'Contrasts_MT_maps',
                              contrasts_name[idx] + '.nii.gz'))

    # Compute and thresold MT maps
    MTR, MTsat = compute_MT_maps(computed_contrasts, parameters)
    for curr_map in MTR, MTsat:
        curr_map = threshold_maps(curr_map, args.in_mask, 0, 100)
        if args.in_B1_map:
            curr_map = apply_B1_correction(curr_map, args.in_B1_map)

    # Save MT maps
    img_name = ['MTR', 'MTsat']

    if args.in_B1_map:
        img_name = [curr_name + '_B1_corrected'
                    for curr_name in img_name]

    if args.out_prefix:
        img_name = [args.out_prefix + '_' + curr_name
                    for curr_name in img_name]

    img_data = MTR, MTsat
    for img_to_save, name in zip(img_data, img_name):
        nib.save(nib.Nifti1Image(img_to_save.astype(np.float32),
                                 ref_img.affine),
                 os.path.join(args.out_dir, 'MT_native_maps',
                              name + '.nii.gz'))


if __name__ == '__main__':
    main()
