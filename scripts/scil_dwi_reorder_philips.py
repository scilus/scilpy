#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-order gradient according to original table (Philips)
This script is not needed for version 5.6 and higher

Formerly: scil_reorder_dwi_philips.py
"""

import argparse
import json
import logging
from packaging import version
import sys

from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.gradients.utils import get_new_gtab_order
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.filenames import split_name_with_nii

SOFTWARE_VERSION_MIN = '5.6'


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_dwi',
                   help='Input dwi file.')
    p.add_argument('in_bval',
                   help='Input bval FSL format.')
    p.add_argument('in_bvec',
                   help='Input bvec FSL format.')
    p.add_argument('in_table',
                   help='Original philips table - first line is skipped.')
    p.add_argument('out_basename',
                   help='Basename output file.')

    p.add_argument('--json',
                   help='If you give a json file, it will check if you need '
                        'to reorder your Philips dwi.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    required_args = [args.in_dwi, args.in_bvec, args.in_bval, args.in_table]

    _, extension = split_name_with_nii(args.in_dwi)
    output_filenames = [args.out_basename + extension,
                        args.out_basename + '.bval',
                        args.out_basename + '.bvec']

    assert_inputs_exist(parser, required_args, args.json)
    assert_outputs_exist(parser, args, output_filenames)

    if args.json and not args.overwrite:
        with open(args.json) as curr_json:
            dwi_json = json.load(curr_json)
        if 'SoftwareVersions' in dwi_json.keys():
            curr_version = dwi_json['SoftwareVersions']
            curr_version = curr_version.replace('\\',
                                                ' ').replace('_',
                                                             ' ').split()[0]
            if version.parse(SOFTWARE_VERSION_MIN) <= version.parse(
                    curr_version):
                sys.exit('ERROR: There is no need for reording since your '
                         'dwi comes from a Philips machine with '
                         'version {}. '.format(curr_version) +
                         'No file will be created. \n'
                         'Use -f to force overwriting.')

    philips_table = np.loadtxt(args.in_table, skiprows=1)
    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)
    dwi = nib.load(args.in_dwi)

    new_index = get_new_gtab_order(philips_table, dwi, bvals, bvecs)
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
