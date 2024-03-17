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
are obtained using six contrasts: single positive frequency image, single
negative frequency image, dual alternating positive/negative frequency image,
dual alternating negative/positive frequency image (saturated images);
and two unsaturated contrasts as reference. These two references should be
acquired with predominant PD (proton density) and T1 weighting at different
excitation flip angles (a_PD, a_T1) and repetition times (TR_PD, TR_T1).


Input Data recommendation:
  - it is recommended to use dcm2niix (v1.0.20200331) to convert data
    https://github.com/rordenlab/dcm2niix/releases/tag/v1.0.20200331
  - dcm2niix conversion will create all echo files for each contrast and
    corresponding json files
  - all contrasts must have a same number of echoes and coregistered
    between them before running the script
  - Mask must be coregistered to the echo images
  - ANTs can be used for the registration steps (http://stnava.github.io/ANTs/)


The output consists of a ihMT_native_maps folder containing the 4 myelin maps:
    - MTR.nii.gz : Magnetization Transfer Ratio map
    - ihMTR.nii.gz : inhomogeneous Magnetization Transfer Ratio map
    The (ih)MT ratio is a measure reflecting the amount of bound protons.
    - MTsat.nii.gz : Magnetization Transfer saturation map
    - ihMTsat.nii.gz : inhomogeneous Magnetization Transfer saturation map
    The (ih)MT saturation is a pseudo-quantitative maps representing
    the signal change between the bound and free water pools.

As an option, the Complementary_maps folder contains the following images:
    - altnp.nii.gz : dual alternating negative and positive frequency image
    - altpn.nii.gz : dual alternating positive and negative frequency image
    - positive.nii.gz : single positive frequency image
    - negative.nii.gz : single negative frequency image
    - mtoff_PD.nii.gz : unsaturated proton density weighted image
    - mtoff_T1.nii.gz : unsaturated T1 weighted image
    - MTsat_d.nii.gz : MTsat computed from the mean dual frequency images
    - MTsat_sp.nii.gz : MTsat computed from the single positive frequency image
    - MTsat_sn.nii.gz : MTsat computed from the single negative frequency image
    - R1app.nii.gz : Apparent R1 map computed for MTsat.
    - B1_map.nii.gz : B1 map after correction and smoothing (if given).


The final maps from ihMT_native_maps can be corrected for B1+ field
  inhomogeneity, using either an empiric method with
  --in_B1_map option, suffix *B1_corrected is added for each map.
  --B1_correction_method empiric
  or a model-based method with
  --in_B1_map option, suffix *B1_corrected is added for each map.
  --B1_correction_method model_based
  --B1_fitValues 3 .mat files, obtained externally from
    https://github.com/TardifLab/OptimizeIHMTimaging/tree/master/b1Correction,
    and given in this order: positive frequency saturation, negative frequency
    saturation, dual frequency saturation.
For both methods, the nominal value of the B1 map can be set with
  --B1_nominal value


>>> scil_mti_maps_ihMT.py path/to/output/directory
    --in_altnp path/to/echo*altnp.nii.gz --in_altpn path/to/echo*altpn.nii.gz
    --in_mtoff_pd path/to/echo*mtoff.nii.gz
    --in_negative path/to/echo*neg.nii.gz --in_positive path/to/echo*pos.nii.gz
    --in_mtoff_t1 path/to/echo*T1w.nii.gz --mask path/to/mask_bin.nii.gz
    --in_jsons path/to/echo*mtoff.json path/to/echo*T1w.json

By default, the script uses all the echoes available in the input folder.
If you want to use a single echo, replace the * with the specific number of
the echo.
"""

import argparse
import logging
import os

import nibabel as nib
import numpy as np

from scilpy.io.mti import add_common_args_mti, load_and_verify_mti
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist, add_verbose_arg,
                             assert_output_dirs_exist_and_empty)
from scilpy.reconst.mti import (apply_B1_corr_empiric,
                                apply_B1_corr_model_based,
                                compute_ratio_map,
                                compute_saturation_map,
                                threshold_map)

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
    p.add_argument('--out_prefix',
                   help='Prefix to be used for each output image.')
    p.add_argument('--mask',
                   help='Path to the binary brain mask.')

    g = p.add_argument_group(title='Contrast maps', description='Path to '
                             'echoes corresponding to contrast images. All '
                             'constrasts must have \nthe same number of '
                             'echoes and coregistered between them. '
                             'Use * to include all echoes.')
    g.add_argument('--in_altnp', nargs='+', required=True,
                   help='Path to all echoes corresponding to the '
                        'alternation of \nnegative and positive '
                        'frequency saturation pulse.')
    g.add_argument('--in_altpn', nargs='+', required=True,
                   help='Path to all echoes corresponding to the '
                        'alternation of \npositive and negative '
                        'frequency saturation pulse.')
    g.add_argument("--in_negative", nargs='+', required=True,
                   help='Path to all echoes corresponding to the '
                        'negative frequency \nsaturation pulse.')
    g.add_argument("--in_positive", nargs='+', required=True,
                   help='Path to all echoes corresponding to the '
                        'positive frequency \nsaturation pulse.')
    g.add_argument("--in_mtoff_pd", nargs='+', required=True,
                   help='Path to all echoes corresponding to the predominant '
                        'PD \n(proton density) weighting images with no '
                        'saturation pulse.')
    g.add_argument("--in_mtoff_t1", nargs='+',
                   help='Path to all echoes corresponding to the predominant '
                        'T1 \nweighting images with no saturation pulse. This '
                        'one is optional, \nsince it is only needed for the '
                        'calculation of MTsat and ihMTsat. \nAcquisition '
                        'parameters should also be set with this image.')

    # Other MTI arguments are gathered here.
    add_common_args_mti(p)

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    output_dir = os.path.join(args.out_dir, 'ihMT_native_maps')
    extended_dir = None
    if args.extended:
        extended_dir = os.path.join(args.out_dir, 'Complementary_maps')
        assert_output_dirs_exist_and_empty(parser, args, extended_dir,
                                           output_dir, create_dir=True)
    else:
        assert_output_dirs_exist_and_empty(parser, args, output_dir,
                                           create_dir=True)

    if args.out_prefix:
        out_prefix = args.out_prefix + "_"
    else:
        out_prefix = ""

    # Combine all echos path into a list of lists
    input_maps_lists = [args.in_altnp, args.in_altpn, args.in_negative,
                        args.in_positive, args.in_mtoff_pd]
    if args.in_mtoff_t1:
        input_maps_lists.append(args.in_mtoff_t1)

    input_maps_flat_list = [m for _list in input_maps_lists for m in _list]
    assert_inputs_exist(parser, input_maps_flat_list,
                        optional=args.in_mtoff_t1 or [] + [args.mask])

    # Define affine. Uses the first in_mtoff_pd (required).
    affine = nib.load(input_maps_lists[4][0]).affine

    # Define contrasts maps names
    contrast_names = ['altnp', 'altpn', 'negative', 'positive', 'mtoff_PD',
                      'mtoff_T1']

    # Other checks, loading, saving contrast_maps.
    single_echo, flip_angles, rep_times, B1_map, contrast_maps = \
        load_and_verify_mti(args, parser, input_maps_lists, extended_dir,
                            affine, contrast_names)

    # Compute ratio maps
    MTR, ihMTR = compute_ratio_map((contrast_maps[2] + contrast_maps[3]) / 2,
                                   contrast_maps[4],
                                   mt_on_dual=(contrast_maps[0] +
                                               contrast_maps[1]) / 2)
    img_name = ['ihMTR', 'MTR']
    img_data = [ihMTR, MTR]

    # Compute saturation maps
    if args.in_mtoff_t1:
        MTsat_sp, T1app = compute_saturation_map(contrast_maps[3],
                                                 contrast_maps[4],
                                                 contrast_maps[5],
                                                 flip_angles, rep_times)
        MTsat_sn, _ = compute_saturation_map(contrast_maps[2],
                                             contrast_maps[4],
                                             contrast_maps[5],
                                             flip_angles, rep_times)
        MTsat_d, _ = compute_saturation_map((contrast_maps[0] +
                                             contrast_maps[1]) / 2,
                                            contrast_maps[4], contrast_maps[5],
                                            flip_angles, rep_times)
        R1app = 1000 / T1app  # convert 1/ms to 1/s
        if args.extended:
            nib.save(nib.Nifti1Image(MTsat_sp, affine),
                     os.path.join(extended_dir,
                                  out_prefix + "MTsat_single_positive.nii.gz"))
            nib.save(nib.Nifti1Image(MTsat_sn, affine),
                     os.path.join(extended_dir,
                                  out_prefix + "MTsat_single_negative.nii.gz"))
            nib.save(nib.Nifti1Image(MTsat_d, affine),
                     os.path.join(extended_dir,
                                  out_prefix + "MTsat_dual.nii.gz"))
            nib.save(nib.Nifti1Image(R1app, affine),
                     os.path.join(extended_dir,
                                  out_prefix + "apparent_R1.nii.gz"))

        MTsat_maps = [MTsat_sp, MTsat_sn, MTsat_d]

        # Apply model-based B1 correction
        if args.in_B1_map and args.B1_correction_method == 'model_based':
            for i, MTsat_map in enumerate(MTsat_maps):
                MTsat_maps[i] = apply_B1_corr_model_based(MTsat_map, B1_map,
                                                          R1app,
                                                          args.B1_fitvalues[i])

        # Compute MTsat and ihMTsat from saturations
        MTsat = (MTsat_maps[0] + MTsat_maps[1]) / 2
        ihMTsat = MTsat_maps[2] - MTsat

        # Apply empiric B1 correction
        if args.in_B1_map and args.B1_correction_method == 'empiric':
            MTsat = apply_B1_corr_empiric(MTsat, B1_map)
            ihMTsat = apply_B1_corr_empiric(ihMTsat, B1_map)

        img_name.extend(('ihMTsat', 'MTsat'))
        img_data.extend((ihMTsat, MTsat))

    # Apply thresholds on maps
    upper_thresholds = [100, 100, 10, 10]
    idx_contrast_lists = [[0, 1, 2, 3, 4], [3, 4], [0, 1, 2, 3], [3, 4]]
    for i, map in enumerate(img_data):
        img_data[i] = threshold_map(map, args.mask, 0, upper_thresholds[i],
                                    idx_contrast_list=idx_contrast_lists[i],
                                    contrast_maps=contrast_maps)

    # Save ihMT and MT images
    if args.filtering:
        img_name = [curr_name + '_filter'
                    for curr_name in img_name]
    if single_echo:
        img_name = [curr_name + '_single_echo'
                    for curr_name in img_name]
    if args.in_B1_map:
        img_name = [curr_name + '_B1_corrected'
                    for curr_name in img_name]
    if args.out_prefix:
        img_name = [args.out_prefix + '_' + curr_name
                    for curr_name in img_name]

    for img_to_save, name in zip(img_data, img_name):
        nib.save(nib.Nifti1Image(img_to_save.astype(np.float32), affine),
                 os.path.join(output_dir, name + '.nii.gz'))


if __name__ == '__main__':
    main()
