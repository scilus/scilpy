#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-order gradient according to original table
"""

import argparse
import logging

from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.gradients.utils import get_new_order_philips
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.filenames import split_name_with_nii


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_dwi',
                   help='Input dwi file.')
    p.add_argument('in_bvec',
                   help='Input bvec FSL format.')
    p.add_argument('in_bval',
                   help='Input bval FSL format.')
    p.add_argument('in_table',
                   help='Original philips table - first line is skipped.')
    p.add_argument('out_basename',
                   help='Basename output file.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    required_args = [args.in_dwi, args.in_bvec, args.in_bval, args.in_table]

    _, extension = split_name_with_nii(args.in_dwi)
    output_filenames = [args.out_basename + extension,
                        args.out_basename + '.bval',
                        args.out_basename + '.bvec']

    assert_inputs_exist(parser, required_args)
    assert_outputs_exist(parser, args, output_filenames)

    philips_table = np.loadtxt(args.in_table, skiprows=1)
    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)
    dwi = nib.load(args.in_dwi)

    new_index = get_new_order_philips(philips_table, dwi, bvals, bvecs)
    bvecs = bvecs[new_index]
    bvals = bvals[new_index]

    data = dwi.dataobj.get_unscaled()
    data = data[:, :, :, new_index]

    tmp = nib.Nifti1Image(data, dwi.affine, header=dwi.header)
    tmp.header['scl_slope'] = dwi.dataobj.slope
    tmp.header['scl_inter'] = dwi.dataobj.inter
    tmp.update_header()

    nib.save(tmp, output_filenames[0])
    np.savetxt(args.out_basename + '.bval', bvals.reshape(1, len(bvals)), '%d')
    np.savetxt(args.out_basename + '.bvec', bvecs.T, '%0.15f')


if __name__ == '__main__':
    main()
