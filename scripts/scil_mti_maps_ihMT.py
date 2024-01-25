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

>>> scil_mti_maps_ihMT.py path/to/output/directory path/to/mask_bin.nii.gz
    --in_altnp path/to/echo*altnp.nii.gz --in_altpn path/to/echo*altpn.nii.gz
    --in_mtoff path/to/echo*mtoff.nii.gz --in_negative path/to/echo*neg.nii.gz
    --in_positive path/to/echo*pos.nii.gz --in_t1w path/to/echo*T1w.nii.gz

By default, the script uses all the echoes available in the input folder.
If you want to use a single echo add --single_echo to the command line and
replace the * with the specific number of the echo.

Formerly: scil_compute_ihMT_maps.py
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
                                compute_ihMT_maps, threshold_maps,
                                compute_MT_maps_from_ihMT,
                                apply_B1_correction)

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

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


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
    parameters = []
    for curr_map in maps[4][0], maps[5][0]:
        acq_parameter = get_acq_parameters(curr_map.replace('.nii.gz',
                                                            '.json'),
                                           ['RepetitionTime', 'FlipAngle'])
        acq_parameter = acq_parameter[0]*1000, acq_parameter[1]*np.pi/180
        parameters.append(acq_parameter)

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
        input_images = []
        for image in curr_map:
            img, _ = load_img(image)
            input_images.append(img)
        merged_curr_map = concatenate(input_images, input_images[0])
        computed_contrasts.append(compute_contrasts_maps(
                                  merged_curr_map, filtering=args.filtering,
                                  single_echo=args.single_echo))

        nib.save(nib.Nifti1Image(computed_contrasts[idx].astype(np.float32),
                                 ref_img.affine),
                 os.path.join(args.out_dir, 'Contrasts_ihMT_maps',
                              contrasts_name[idx] + '.nii.gz'))

    # Compute and thresold ihMT maps
    ihMTR, ihMTsat = compute_ihMT_maps(computed_contrasts, parameters)
    ihMTR = threshold_maps(ihMTR, args.in_mask, 0, 100,
                           idx_contrast_list=[4, 3, 1, 0, 2],
                           contrasts_maps=computed_contrasts)
    ihMTsat = threshold_maps(ihMTsat, args.in_mask, 0, 10,
                             idx_contrast_list=[4, 3, 1, 0],
                             contrasts_maps=computed_contrasts)
    if args.in_B1_map:
        ihMTR = apply_B1_correction(ihMTR, args.in_B1_map)
        ihMTsat = apply_B1_correction(ihMTsat, args.in_B1_map)

    # Compute and thresold non-ihMT maps
    MTR, MTsat = compute_MT_maps_from_ihMT(computed_contrasts, parameters)
    for curr_map in MTR, MTsat:
        curr_map = threshold_maps(curr_map, args.in_mask, 0, 100,
                                  idx_contrast_list=[4, 2],
                                  contrasts_maps=computed_contrasts)
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
