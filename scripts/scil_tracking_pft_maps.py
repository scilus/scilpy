#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute include and exclude maps, and the seeding interface mask from partial
volume estimation (PVE) maps. Maps should have values in [0,1], gm+wm+csf=1 in
all voxels of the brain, gm+wm+csf=0 elsewhere.

References: Girard, G., Whittingstall K., Deriche, R., and Descoteaux, M.
(2014). Towards quantitative connectivity analysis: reducing tractography
biases. Neuroimage.

Formerly: scil_compute_maps_for_particle_filter_tracking.py
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
    p.add_argument('in_wm',
                   help='White matter PVE map (nifti). From normal FAST '
                        'output, has a PVE_2 name suffix.')
    p.add_argument('in_gm',
                   help='Grey matter PVE map (nifti). From normal FAST output,'
                        ' has a PVE_1 name suffix.')
    p.add_argument('in_csf',
                   help='Cerebrospinal fluid PVE map (nifti). From normal FAST'
                        ' output, has a PVE_0 name suffix.')

    p.add_argument('--include', metavar='filename',
                   default='map_include.nii.gz',
                   help='Output include map (nifti). [%(default)s]')
    p.add_argument('--exclude', metavar='filename',
                   default='map_exclude.nii.gz',
                   help='Output exclude map (nifti). [%(default)s]')
    p.add_argument('--interface', metavar='filename',
                   default='interface.nii.gz',
                   help='Output interface seeding mask (nifti). [%(default)s]')
    p.add_argument('-t', dest='int_thres', metavar='THRESHOLD',
                   type=float, default=0.1,
                   help='Minimum gm and wm PVE values in a voxel to be into '
                        'the interface. [%(default)s]')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_wm, args.in_gm, args.in_csf])
    assert_outputs_exist(parser, args,
                         [args.include, args.exclude, args.interface])

    wm_pve = nib.load(args.in_wm)
    gm_pve = nib.load(args.in_gm)
    csf_pve = nib.load(args.in_csf)

    background = np.ones(gm_pve.shape)
    background[gm_pve.get_fdata(dtype=np.float32) > 0] = 0
    background[wm_pve.get_fdata(dtype=np.float32) > 0] = 0
    background[csf_pve.get_fdata(dtype=np.float32) > 0] = 0

    include_map = gm_pve.get_fdata(dtype=np.float32)
    include_map[background > 0] = 1

    exclude_map = csf_pve.get_fdata(dtype=np.float32)

    interface = np.zeros(gm_pve.shape, dtype=np.uint8)
    interface[gm_pve.get_fdata(dtype=np.float32) >= args.int_thres] = 1
    interface[wm_pve.get_fdata(dtype=np.float32) < args.int_thres] = 0

    logging.info('The interface "{0}" contains {1} voxels.'.format(
        args.interface, int(np.sum(interface))))

    nib.save(nib.Nifti1Image(include_map, gm_pve.affine),
             args.include)

    nib.save(nib.Nifti1Image(exclude_map, gm_pve.affine),
             args.exclude)

    nib.save(nib.Nifti1Image(interface, gm_pve.affine),
             args.interface)


if __name__ == "__main__":
    main()
