#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the axial (para_diff), radial (perp_diff), and mean (iso_diff)
diffusivity priors for NODDI.

Formerly: scil_compute_NODDI_priors.py
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.image import assert_same_resolution
from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_verbose_arg)

EPILOG = """
Reference:
    [1] Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC.
        NODDI: practical in vivo neurite orientation dispersion and density
        imaging of the human brain. NeuroImage. 2012 Jul 16;61:1000-16.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, epilog=EPILOG,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_FA',
                   help='Path to the FA volume.')
    p.add_argument('in_AD',
                   help='Path to the axial diffusivity (AD) volume.')
    p.add_argument('in_RD',
                   help='Path to the radial diffusivity (RD) volume.')
    p.add_argument('in_MD',
                   help='Path to the mean diffusivity (MD) volume.')

    g1 = p.add_argument_group('Metrics options')
    g1.add_argument(
        '--fa_min_single_fiber', type=float, default=0.7,
        help='Minimal threshold of FA (voxels above that threshold are '
             'considered in \nthe single fiber mask). [%(default)s]')
    g1.add_argument(
        '--fa_max_ventricles', type=float, default=0.1,
        help='Maximal threshold of FA (voxels under that threshold are '
             'considered in \nthe ventricles). [%(default)s]')
    g1.add_argument(
        '--md_min_ventricles', type=float, default=0.003,
        help='Minimal threshold of MD in mm2/s (voxels above that threshold '
             'are considered \nfor in the ventricles). [%(default)s]')

    g2 = p.add_argument_group('Regions options')
    g2.add_argument(
        '--roi_radius', type=int, default=20,
        help='Radius of the region used to estimate the priors. The roi will '
             'be a cube spanning \nfrom ROI_CENTER in each direction. '
             '[%(default)s]')
    g2.add_argument(
        '--roi_center', metavar='pos', nargs=3, type=int,
        help='Center of the roi of size roi_radius used to estimate the '
             'priors; a 3-value coordinate. \nIf not set, uses the center of '
             'the 3D volume.')

    g3 = p.add_argument_group('Outputs')
    g3.add_argument('--out_txt_1fiber_para', metavar='FILE',
                    help='Output path for the text file containing the single '
                         'fiber average value of AD.\nIf not set, the file '
                         'will not be saved.')
    g3.add_argument('--out_txt_1fiber_perp', metavar='FILE',
                    help='Output path for the text file containing the single '
                         'fiber average value of RD.\nIf not set, the file '
                         'will not be saved.')
    g3.add_argument('--out_mask_1fiber', metavar='FILE',
                    help='Output path for single fiber mask. If not set, the '
                         'mask will not be saved.')
    g3.add_argument('--out_txt_ventricles', metavar='FILE',
                    help='Output path for the text file containing the '
                         'ventricles average value of MD.\nIf not set, the '
                         'file will not be saved.')
    g3.add_argument('--out_mask_ventricles', metavar='FILE',
                    help='Output path for the ventricule mask.\nIf not set, '
                         'the mask will not be saved.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    assert_inputs_exist(parser, [args.in_AD, args.in_FA, args.in_MD,
                                 args.in_RD])
    assert_outputs_exist(parser, args, [],
                         [args.out_mask_1fiber,
                          args.out_mask_ventricles,
                          args.out_txt_ventricles,
                          args.out_txt_1fiber_para,
                          args.out_txt_1fiber_perp])

    assert_same_resolution([args.in_AD, args.in_FA, args.in_MD, args.in_RD])

    # Loading
    fa_img = nib.load(args.in_FA)
    fa_data = fa_img.get_fdata(dtype=np.float32)
    affine = fa_img.affine

    md_data = nib.load(args.in_MD).get_fdata(dtype=np.float32)
    ad_data = nib.load(args.in_AD).get_fdata(dtype=np.float32)
    rd_data = nib.load(args.in_RD).get_fdata(dtype=np.float32)

    # Finding ROI center
    if args.roi_center is None:
        # Using the center of the image: single-fiber region should be the CC
        ci, cj, ck = np.array(fa_data.shape[:3]) // 2
    else:
        if len(args.roi_center) != 3:
            parser.error("roi_center needs to receive 3 values")
        elif not np.all(np.asarray(args.roi_center) > 0):
            parser.error("roi_center needs to be positive")
        else:
            ci, cj, ck = args.roi_center

    # Get values in the ROI
    w = args.roi_radius
    roi_posx = slice(max(int(ci - w), 0), min(int(ci + w), fa_data.shape[0]))
    roi_posy = slice(max(int(cj - w), 0), min(int(cj + w), fa_data.shape[1]))
    roi_posz = slice(max(int(ck - w), 0), min(int(ck + w), fa_data.shape[2]))
    roi_ad = ad_data[roi_posx, roi_posy, roi_posz]
    roi_rd = rd_data[roi_posx, roi_posy, roi_posz]
    roi_md = md_data[roi_posx, roi_posy, roi_posz]
    roi_fa = fa_data[roi_posx, roi_posy, roi_posz]

    # Get information in single fiber voxels
    # Taking voxels with FA < 0.95 just to avoid using weird broken voxels.
    indices = np.where((roi_fa > args.fa_min_single_fiber) & (roi_fa < 0.95))
    nb_voxels = roi_fa[indices].shape[0]
    logging.info('Number of voxels found in single fiber area (FA in range '
                 '{}-{}]: {}'
                 .format(args.fa_min_single_fiber, 0.95, nb_voxels))
    single_fiber_ad_mean = np.mean(roi_ad[indices])
    single_fiber_ad_std = np.std(roi_ad[indices])
    single_fiber_rd_mean = np.mean(roi_rd[indices])
    single_fiber_rd_std = np.std(roi_rd[indices])

    # Create mask of single fiber in ROI
    indices[0][:] += ci - w
    indices[1][:] += cj - w
    indices[2][:] += ck - w
    mask_single_fiber = np.zeros(fa_data.shape, dtype=np.uint8)
    mask_single_fiber[indices] = 1

    # Get information in ventricles
    indices = np.where((roi_md > args.md_min_ventricles) &
                       (roi_fa < args.fa_max_ventricles))
    nb_voxels = roi_md[indices].shape[0]
    logging.info('Number of voxels found in ventricles (FA < {} and MD > {}): '
                 '{}'.format(args.fa_max_ventricles, args.md_min_ventricles,
                             nb_voxels))

    vent_avg = np.mean(roi_md[indices])
    vent_std = np.std(roi_md[indices])

    # Create mask of ventricle in ROI
    indices[0][:] += ci - w
    indices[1][:] += cj - w
    indices[2][:] += ck - w
    mask_vent = np.zeros(fa_data.shape, dtype=np.uint8)
    mask_vent[indices] = 1

    # Saving
    if args.out_mask_1fiber:
        nib.save(nib.Nifti1Image(mask_single_fiber, affine),
                 args.out_mask_1fiber)

    if args.out_mask_ventricles:
        nib.save(nib.Nifti1Image(mask_vent, affine), args.out_mask_ventricles)

    if args.out_txt_1fiber_para:
        np.savetxt(args.out_txt_1fiber_para, [single_fiber_ad_mean], fmt='%f')

    if args.out_txt_1fiber_perp:
        np.savetxt(args.out_txt_1fiber_perp, [single_fiber_rd_mean], fmt='%f')

    if args.out_txt_ventricles:
        np.savetxt(args.out_txt_ventricles, [vent_avg], fmt='%f')

    logging.info("Average AD in single fiber areas: {} +- {}"
                 .format(single_fiber_ad_mean, single_fiber_ad_std))
    logging.info("Average RD in single fiber areas: {} +- {}"
                 .format(single_fiber_rd_mean, single_fiber_rd_std))
    logging.info("Average MD in ventricles: {} +- {}"
                 .format(vent_avg, vent_std))


if __name__ == "__main__":
    main()
