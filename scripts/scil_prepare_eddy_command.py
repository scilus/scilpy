#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare a typical command for eddy and create the necessary files.
"""

import argparse
import logging
import os
import subprocess

from dipy.io.gradients import read_bvals_bvecs
import numpy as np
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist)
from scilpy.preprocessing.distortion_correction import (create_acqparams,
                                                        create_index,
                                                        create_non_zero_norm_bvecs)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_dwi',
                   help='input DWI Nifti image')

    p.add_argument('in_bvals',
                   help='b-values file in FSL format')

    p.add_argument('in_bvecs',
                   help='b-vectors file in FSL format')

    p.add_argument('in_mask',
                   help='binary brain mask.')

    p.add_argument('--topup',
                   help='topup output name. ' +
                        'If given, apply topup during eddy.\n' +
                        'Should be the same as --out_prefix from ' +
                        'scil_prepare_topup_command.py')

    p.add_argument('--eddy_cmd', default='eddy_openmp',
                   choices=['eddy_openmp', 'eddy_cuda'],
                   help='eddy command [%(default)s].')

    p.add_argument('--b0_thr', type=float, default=20,
                   help='All b-values with values less than or equal ' +
                        'to b0_thr are considered\nas b0s i.e. without ' +
                        'diffusion weighting')

    p.add_argument('--encoding_direction', default='y',
                   choices=['x', 'y', 'z'],
                   help='acquisition direction, default is AP-PA '
                        '[%(default)s].')

    p.add_argument('--readout', type=float, default=0.062,
                   help='total readout time from the DICOM metadata '
                        '[%(default)s].')

    p.add_argument('--slice_drop_correction', action='store_true',
                   help="if set, will activate eddy's outlier correction,\n"
                        "which includes slice drop correction.")

    p.add_argument('--out_directory', default='.',
                   help='output directory for eddy files [%(default)s].')

    p.add_argument('--out_prefix', default='dwi_eddy_corrected',
                   help='prefix of the eddy-corrected DWI [%(default)s].')

    p.add_argument('--out_script', action='store_true',
                   help='if set, will output a .sh script (eddy.sh).\n' +
                        'else, will output the lines to the ' +
                        'terminal [%(default)s].')

    p.add_argument('--fix_seed', action='store_true',
                   help='if set, will use the fixed seed strategy for eddy.\n'
                        'Enhances reproducibility.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        devnull = open(os.devnull)
        subprocess.call(args.eddy_cmd, stderr=devnull)
    except:
        parser.error("Please download the {} command.".format(args.eddy_cmd))

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    required_args = [args.in_dwi, args.in_bvals, args.in_bvecs, args.in_mask]

    assert_inputs_exist(parser, required_args)

    if os.path.splitext(args.out_prefix)[1] != '':
        parser.error('The prefix must not contain any extension.')

    bvals, bvecs = read_bvals_bvecs(args.in_bvals, args.in_bvecs)
    bvals_min = bvals.min()
    b0_threshold = args.b0_thr
    if bvals_min < 0 or bvals_min > b0_threshold:
        raise ValueError('The minimal b-value is lesser than 0 or greater '
                         'than {0}. This is highly suspicious. Please check '
                         'your data to ensure everything is correct. '
                         'Value found: {1}'.format(b0_threshold, bvals_min))

    acqparams = create_acqparams(args.readout, args.encoding_direction,
                                 nb_rev_b0s=0)
    index = create_index(bvals)
    bvecs = create_non_zero_norm_bvecs(bvecs)

    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    acqparams_path = os.path.join(args.out_directory, 'acqparams.txt')
    np.savetxt(acqparams_path, acqparams, fmt='%1.4f', delimiter=' ')
    bvecs_path = os.path.join(args.out_directory, 'non_zero_norm.bvecs')
    np.savetxt(bvecs_path, bvecs.T, fmt="%.8f")
    index_path = os.path.join(args.out_directory, 'index.txt')
    np.savetxt(index_path, index, fmt='%i', newline=" ")

    additional_args = ""
    if args.topup is not None:
        additional_args += "--topup={} ".format(args.topup)

    if args.slice_drop_correction:
        additional_args += "--repol "

    if args.fix_seed:
        additional_args += "--initrand "

    output_path = os.path.join(args.out_directory, args.out_prefix)
    eddy = '{0} --imain={1} --mask={2} --acqp={3} --index={4}' \
           ' --bvecs={5} --bvals={6} --out={7} --data_is_shelled {8}' \
        .format(args.eddy_cmd, args.in_dwi, args.in_mask, acqparams_path,
                index_path, bvecs_path, args.in_bvals, output_path,
                additional_args)

    if args.out_script:
        with open("eddy.sh", 'w') as f:
            f.write(eddy)
    else:
        print(eddy)


if __name__ == '__main__':
    main()
