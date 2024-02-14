#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the axial (para_diff), radial (perp_diff),
and mean (iso_diff) diffusivity priors for NODDI.

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
        NODDI: practical in vivo neurite orientation dispersion
        and density imaging of the human brain.
        NeuroImage. 2012 Jul 16;61:1000-16.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
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
        '--fa_min', type=float, default='0.7',
        help='Minimal threshold of FA (voxels above that threshold are '
             'considered in the single fiber mask). [%(default)s]')
    g1.add_argument(
        '--fa_max', type=float, default='0.1',
        help='Maximal threshold of FA (voxels under that threshold are '
             'considered in the ventricles). [%(default)s]')
    g1.add_argument(
        '--md_min', dest='md_min',  type=float, default='0.003',
        help='Minimal threshold of MD in mm2/s (voxels above that threshold '
             'are considered for in the ventricles). [%(default)s]')

    g2 = p.add_argument_group('Regions options')
    g2.add_argument(
        '--roi_radius', type=int, default=20,
        help='Radius of the region used to estimate the priors. The roi will '
             'be a cube spanning from ROI_CENTER in each direction. '
             '[%(default)s]')
    g2.add_argument(
        '--roi_center', metavar='tuple(3)', nargs="+", type=int,
        help='Center of the roi of size roi_radius used to estimate the '
             'priors. [center of the 3D volume]')

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

    assert_inputs_exist(parser, [args.in_AD, args.in_FA, args.in_MD,
                                 args.in_RD])
    assert_outputs_exist(parser, args, [],
                         [args.out_mask_1fiber,
                          args.out_mask_ventricles,
                          args.out_txt_ventricles,
                          args.out_txt_1fiber_para,
                          args.out_txt_1fiber_perp])

    assert_same_resolution([args.in_AD, args.in_FA, args.in_MD, args.in_RD])

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_level)

    fa_img = nib.load(args.in_FA)
    fa_data = fa_img.get_fdata(dtype=np.float32)
    affine = fa_img.affine

    md_data = nib.load(args.in_MD).get_fdata(dtype=np.float32)
    ad_data = nib.load(args.in_AD).get_fdata(dtype=np.float32)
    rd_data = nib.load(args.in_RD).get_fdata(dtype=np.float32)

    mask_cc = np.zeros(fa_data.shape, dtype=np.uint8)
    mask_vent = np.zeros(fa_data.shape, dtype=np.uint8)

    # center
    if args.roi_center is None:
        ci, cj, ck = np.array(fa_data.shape[:3]) // 2
    else:
        if len(args.roi_center) != 3:
            parser.error("roi_center needs to receive 3 values")
        elif not np.all(np.asarray(args.roi_center) > 0):
            parser.error("roi_center needs to be positive")
        else:
            ci, cj, ck = args.roi_center

    w = args.roi_radius
    fa_shape = fa_data.shape
    roi_ad = ad_data[max(int(ci - w), 0): min(int(ci + w), fa_shape[0]),
                     max(int(cj - w), 0): min(int(cj + w), fa_shape[1]),
                     max(int(ck - w), 0): min(int(ck + w), fa_shape[2])]
    roi_rd = rd_data[max(int(ci - w), 0): min(int(ci + w), fa_shape[0]),
                     max(int(cj - w), 0): min(int(cj + w), fa_shape[1]),
                     max(int(ck - w), 0): min(int(ck + w), fa_shape[2])]
    roi_md = md_data[max(int(ci - w), 0): min(int(ci + w), fa_shape[0]),
                     max(int(cj - w), 0): min(int(cj + w), fa_shape[1]),
                     max(int(ck - w), 0): min(int(ck + w), fa_shape[2])]
    roi_fa = fa_data[max(int(ci - w), 0): min(int(ci + w), fa_shape[0]),
                     max(int(cj - w), 0): min(int(cj + w), fa_shape[1]),
                     max(int(ck - w), 0): min(int(ck + w), fa_shape[2])]

    logging.debug('fa_min, fa_max, md_min: {}, {}, {}'.format(
        args.fa_min, args.fa_max, args.md_min))

    indices = np.where((roi_fa > args.fa_min) & (roi_fa < 0.95))
    N = roi_ad[indices].shape[0]

    logging.debug('Number of voxels found in single fiber area: {}'.format(N))

    cc_avg_para = np.mean(roi_ad[indices])
    cc_std_para = np.std(roi_ad[indices])

    cc_avg_perp = np.mean(roi_rd[indices])
    cc_std_perp = np.std(roi_rd[indices])

    indices[0][:] += ci - w
    indices[1][:] += cj - w
    indices[2][:] += ck - w
    mask_cc[indices] = 1

    indices = np.where((roi_md > args.md_min) & (roi_fa < args.fa_max))
    N = roi_md[indices].shape[0]

    logging.debug('Number of voxels found in ventricles: {}'.format(N))

    vent_avg = np.mean(roi_md[indices])
    vent_std = np.std(roi_md[indices])

    indices[0][:] += ci - w
    indices[1][:] += cj - w
    indices[2][:] += ck - w
    mask_vent[indices] = 1

    if args.out_mask_1fiber:
        nib.save(nib.Nifti1Image(mask_cc, affine), args.out_mask_1fiber)

    if args.out_mask_ventricles:
        nib.save(nib.Nifti1Image(mask_vent, affine), args.out_mask_ventricles)

    if args.out_txt_1fiber_para:
        np.savetxt(args.out_txt_1fiber_para, [cc_avg_para], fmt='%f')

    if args.out_txt_1fiber_perp:
        np.savetxt(args.out_txt_1fiber_perp, [cc_avg_perp], fmt='%f')

    if args.out_txt_ventricles:
        np.savetxt(args.out_txt_ventricles, [vent_avg], fmt='%f')

    logging.info("Average AD in single fiber areas: {} +- {}".format(
                                                                cc_avg_para,
                                                                cc_std_para))
    logging.info("Average RD in single fiber areas: {} +- {}".format(
                                                                cc_avg_perp,
                                                                cc_std_perp))
    logging.info("Average MD in ventricles: {} +- {}".format(vent_avg,
                                                             vent_std))


if __name__ == "__main__":
    main()
