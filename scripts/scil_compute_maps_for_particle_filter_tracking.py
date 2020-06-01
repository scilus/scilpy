#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute include and exclude maps, and the seeding interface mask from partial
volume estimation (PVE) maps. Maps should have values in [0,1], gm+wm+csf=1 in
all voxels of the brain, gm+wm+csf=0 elsewhere.

References: Girard, G., Whittingstall K., Deriche, R., and Descoteaux, M.
(2014). Towards quantitative connectivity analysis: reducing tractography
biases. Neuroimage.
"""

import argparse
import logging

import numpy as np
import nibabel as nib

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_verbose_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        'wm',
        help='White matter PVE map (nifti). From normal FAST output, has a '
             'PVE_2 name suffix.')
    p.add_argument(
        'gm',
        help='Grey matter PVE map (nifti). From normal FAST output, has a '
             'PVE_1 name suffix.')
    p.add_argument(
        'csf',
        help='Cerebrospinal fluid PVE map (nifti). From normal FAST output, '
             'has a PVE_0 name suffix.')

    p.add_argument(
        '--include', metavar='filename', default='map_include.nii.gz',
        help='Output include map (nifti). [map_include.nii.gz]')
    p.add_argument(
        '--exclude', metavar='filename', default='map_exclude.nii.gz',
        help='Output exclude map (nifti). [map_exclude.nii.gz]')
    p.add_argument(
        '--interface', metavar='filename', default='interface.nii.gz',
        help='Output interface seeding mask (nifti). [interface.nii.gz]')
    p.add_argument(
        '-t', dest='int_thres', metavar='THRESHOLD', type=float, default=0.1,
        help='Minimum gm and wm PVE values in a voxel to be in to the '
             'interface. [0.1]')
    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    assert_inputs_exist(parser, [args.wm, args.gm, args.csf])
    assert_outputs_exist(parser, args,
                         [args.include, args.exclude, args.interface])

    wm_pve = nib.load(args.wm)
    logging.info('"{0}" loaded as WM PVE map.'.format(args.wm))
    gm_pve = nib.load(args.gm)
    logging.info('"{0}" loaded as GM PVE map.'.format(args.gm))
    csf_pve = nib.load(args.csf)
    logging.info('"{0}" loaded as CSF PVE map.'.format(args.csf))

    background = np.ones(gm_pve.shape)
    background[gm_pve.get_data() > 0] = 0
    background[wm_pve.get_data() > 0] = 0
    background[csf_pve.get_data() > 0] = 0

    include_map = gm_pve.get_data()
    include_map[background > 0] = 1

    exclude_map = csf_pve.get_data()

    interface = np.zeros(gm_pve.shape)
    interface[gm_pve.get_data() >= args.int_thres] = 1
    interface[wm_pve.get_data() < args.int_thres] = 0

    logging.info('The interface "{0}" contains {1} voxels.'.format(
        args.interface, int(np.sum(interface))))

    nib.Nifti1Image(include_map.astype('float32'),
                    gm_pve.affine).to_filename(args.include)
    nib.Nifti1Image(exclude_map.astype('float32'),
                    gm_pve.affine).to_filename(args.exclude)
    nib.Nifti1Image(interface.astype('float32'),
                    gm_pve.affine).to_filename(args.interface)


if __name__ == "__main__":
    main()
