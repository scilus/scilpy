#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute the axial and mean diffusivity priors for NODDI.
"""

import argparse
import logging

import nibabel as nib
import numpy as np

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
    p.add_argument('fa',
                   help='Path to the FA volume.')
    p.add_argument('ad',
                   help='Path to the axial diffusivity (AD) volume.')
    p.add_argument('md',
                   help='Path to the mean diffusivity (MD) volume.')

    g1 = p.add_argument_group('Metrics options')
    g1.add_argument(
        '--fa_max', type=float, default='0.1',
        help='Maximal threshold of FA (voxels under that threshold are '
             'considered in the ventricles). [%(default)s]')
    g1.add_argument(
        '--fa_min', type=float, default='0.7',
        help='Minimal threshold of FA (voxels above that threshold are '
             'considered in the single fiber mask). [%(default)s]')
    g1.add_argument(
        '--md_min', dest='md_t',  type=float, default='0.003',
        help='Minimal threshold of MD in mm2/s (voxels above that threshold '
             'are considered for in the ventricles). [%(default)s]')


    g2 = p.add_argument_group('Regions options')
    g2.add_argument(
        '--roi_radius', default=20, type=int,
        help='Radius of the region used to estimate the priors. The roi will '
             'be a cube spanning from ROI_CENTER in each direction. '
             '[%(default)s]')
    g2.add_argument(
        '--roi_center', metavar='tuple(3)', nargs="+", type=int,
        help='Center of the roi of size roi_radius used to estimate the '
             'priors. [center of the 3D volume]')

    g3 = p.add_argument_group('Outputs')
    g3.add_argument(
        '--output_1fiber', metavar='file',
        help='Output path for the text file containing the single '
             'fiber average value of AD. If not set, the file will not be '
             'saved.')
    g3.add_argument(
        '--mask_output_1fiber', metavar='file',
        help='Output path for single fiber mask. If not set, the mask will '
             'not be saved.')
    g3.add_argument(
        '--output_ventricles', metavar='file',
        help='Output path for the text file containing the ventricles average '
             'value of MD. If not set, the file will not be saved.')
    g3.add_argument(
        '--mask_output_ventricles', metavar='file',
        help='Output path for the ventricule mask. If not set, the mask will '
             'not be saved.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def load(path):
    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header.get_zooms()[:3]


def save(data, affine, output):
    img = nib.Nifti1Image(np.array(data, 'float32'),  affine)
    nib.save(img, output)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.ad, args.fa, args.md])
    assert_outputs_exist(parser, args, [],
                         [args.mask_output_1fiber,
                         args.mask_output_ventricles,
                         args.output_ventricles,
                         args.output_1fiber])

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    fa, aff, hdr = load(args.fa)
    md, _, _ = load(args.md)
    ad, _, _ = load(args.ad)

    mask_cc = np.zeros(fa.shape)
    mask_vent = np.zeros(fa.shape)

    # center
    if args.roi_center is None:
        ci, cj, ck = np.array(fa.shape[:3]) // 2
    else:
        if len(args.roi_center) != 3:
            parser.error("roi_center needs to receive 3 values")
        elif not np.all(np.asarray(args.roi_center)>0):
            parser.error("roi_center needs to be positive")
        else:
            ci, cj, ck = args.roi_center

    w = args.roi_radius
    square = (np.arange(max(int(ci - w),0), max(int(ci + w),0),
              np.arange(max(int(cj - w),0), max(int(cj + w),0),
              np.arange(max(int(ck - w),0), max(int(ck + w),0))

    roi_ad = ad[square]
    roi_md = md[square]
    roi_fa = fa[square]

    logging.debug('fa_min, fa_max, md_t: {}, {}, {}'.format(
        args.fa_min, args.fa_max, args.md_t))

    indices = np.where((roi_fa > args.fa_min) & (roi_fa < 0.95))
    N = roi_ad[indices].shape[0]

    logging.debug('Number of voxels found in single fiber area: {}'.format(N))

    cc_avg = np.mean(roi_ad[indices])
    cc_std = np.std(roi_ad[indices])

    indices[0][:] += ci - w
    indices[1][:] += cj - w
    indices[2][:] += ck - w
    mask_cc[indices] = 1

    indices = np.where((roi_md > args.md_t) & (roi_fa < args.fa_max))
    N = roi_md[indices].shape[0]

    logging.debug('Number of voxels found in ventricles: {}'.format(N))

    vent_avg = np.mean(roi_md[indices])
    vent_std = np.std(roi_md[indices])

    indices[0][:] += ci - w
    indices[1][:] += cj - w
    indices[2][:] += ck - w
    mask_vent[indices] = 1

    if args.mask_output_ventricles:
        save(mask_vent, aff, args.mask_output_ventricles)

    if args.mask_output_1fiber:
        save(mask_cc, aff, args.mask_output_1fiber)

    if args.output_1fiber:
        with open(args.output_1fiber, "w") as text_file:
            text_file.write(str(cc_avg))

    if args.output_ventricles:
        with open(args.output_ventricles, "w") as text_file:
            text_file.write(str(vent_avg))

    logging.info("Average AD in single fiber areas: {} +- {}".format(cc_avg,
                                                                     cc_std))
    logging.info("Average MD in ventricles: {} +- {}".format(vent_avg,
                                                             vent_std))


if __name__ == "__main__":
    main()
