#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare a typical command for eddy and create the necessary files. When using
multiple acquisitions and/or opposite phase directions, images, b-values and
b-vectors should be merged together using scil_dwi_concatenate.py. If using
topup prior to calling this script, images should be concatenated in the same
order as the b0s used with prepare_topup.

Formerly: scil_prepare_eddy_command.py
"""

import argparse
import logging
import os
import subprocess

from dipy.io.gradients import read_bvals_bvecs
import numpy as np
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_fsl_options_exist,
                             assert_inputs_exist)
from scilpy.preprocessing.distortion_correction import \
    (create_acqparams, create_index, create_multi_topup_index,
     create_non_zero_norm_bvecs)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_dwi',
                   help='Input DWI Nifti image. If using multiple '
                        'acquisition and/or opposite phase directions, please '
                        'merge in the same order as for prepare_topup using '
                        'scil_dwi_concatenate.py.')

    p.add_argument('in_bvals',
                   help='Input b-values file in FSL format.')

    p.add_argument('in_bvecs',
                   help='Input b-vectors file in FSL format.')

    p.add_argument('in_mask',
                   help='Binary brain mask.')

    p.add_argument('--n_reverse', type=int, default=0,
                   help='Number of reverse phase volumes included '
                        'in the DWI image [%(default)s].')

    p.add_argument('--topup',
                   help='Topup output name. ' +
                        'If given, apply topup during eddy.\n' +
                        'Should be the same as --out_prefix from ' +
                        'scil_dwi_prepare_topup_command.py.')

    p.add_argument('--topup_params', default='',
                   help='Parameters file (typically named acqparams) '
                        'used to run topup.')

    p.add_argument('--eddy_cmd', default='eddy_openmp',
                   choices=['eddy_openmp', 'eddy_cuda', 'eddy_cuda8.0',
                            'eddy_cuda9.1', 'eddy_cuda10.2',
                            'eddy', 'eddy_cpu'],
                   help='Eddy command [%(default)s].')

    p.add_argument('--b0_thr', type=float, default=20,
                   help='All b-values with values less than or equal ' +
                        'to b0_thr are considered\nas b0s i.e. without ' +
                        'diffusion weighting [%(default)s].')

    p.add_argument('--encoding_direction', default='y',
                   choices=['x', 'y', 'z'],
                   help='Acquisition direction, default is AP-PA '
                        '[%(default)s].')

    p.add_argument('--readout', type=float, default=0.062,
                   help='Total readout time from the DICOM metadata '
                        '[%(default)s].')

    p.add_argument('--slice_drop_correction', action='store_true',
                   help="If set, will activate eddy's outlier correction,\n"
                        "which includes slice drop correction.")

    p.add_argument('--lsr_resampling', action='store_true',
                   help='Perform least-square resampling, allowing eddy to '
                        'combine forward and reverse phase acquisitions for '
                        'better reconstruction. Only works if directions and '
                        'b-values are identical in both phase direction.')

    p.add_argument('--out_directory', default='.',
                   help='Output directory for eddy files [%(default)s].')

    p.add_argument('--out_prefix', default='dwi_eddy_corrected',
                   help='Prefix of the eddy-corrected DWI [%(default)s].')

    p.add_argument('--out_script', action='store_true',
                   help='If set, will output a .sh script (eddy.sh).\n' +
                        'else, will output the lines to the ' +
                        'terminal [%(default)s].')

    p.add_argument('--fix_seed', action='store_true',
                   help='If set, will use the fixed seed strategy for eddy.\n'
                        'Enhances reproducibility.')

    p.add_argument('--eddy_options',  default='',
                   help='Additional options you want to use to run eddy.\n'
                        'Add these options using quotes (i.e. "--ol_nstd=6'
                        ' --mb=4").')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    try:
        devnull = open(os.devnull)
        subprocess.call(args.eddy_cmd, stderr=devnull)
    except:
        logging.warning(
            "{} not found. If executing locally, please install "
            "the command from the FSL library and make sure it is "
            "available in your path.".format(args.eddy_cmd))

    required_args = [args.in_dwi, args.in_bvals, args.in_bvecs, args.in_mask]

    assert_inputs_exist(parser, required_args)
    assert_fsl_options_exist(parser, args.eddy_options, 'eddy')

    if os.path.splitext(args.out_prefix)[1] != '':
        parser.error('The prefix must not contain any extension.')

    if args.n_reverse and not args.lsr_resampling:
        logging.warning(
            'Multiple reverse phase images supplied, but least-square '
            'resampling is disabled. If the reverse phase image was acquired '
            'with the same sampling as the forward phase image, '
            'use --lsr_resampling for better results from Eddy.')

    bvals, bvecs = read_bvals_bvecs(args.in_bvals, args.in_bvecs)
    bvals_min = bvals.min()
    b0_threshold = args.b0_thr
    if bvals_min < 0 or bvals_min > b0_threshold:
        raise ValueError('The minimal b-value is lesser than 0 or greater '
                         'than {0}. This is highly suspicious. Please check '
                         'your data to ensure everything is correct. '
                         'Value found: {1}'.format(b0_threshold, bvals_min))

    n_rev = args.n_reverse

    if args.topup_params:
        acqparams = np.loadtxt(args.topup_params)

        if acqparams.shape[0] == 2:
            index = create_index(bvals, n_rev=n_rev)
        elif acqparams.shape[0] == np.sum(bvals <= b0_threshold):
            index = create_multi_topup_index(
                bvals, 'none', n_rev, b0_threshold)
        else:
            b_mask = np.ma.array(bvals, mask=[bvals > b0_threshold])
            n_b0_clusters = len(np.ma.clump_unmasked(b_mask[:n_rev])) + \
                len(np.ma.clump_unmasked(b_mask[n_rev:]))
            if acqparams.shape[0] == n_b0_clusters:
                index = create_multi_topup_index(
                    bvals, 'cluster', n_rev, b0_threshold)
            else:
                raise ValueError('Could not determine a valid index file '
                                 'from the provided acquisition parameters '
                                 'file: {}'.format(args.topup_params))
    else:
        acqparams = create_acqparams(args.readout, args.encoding_direction,
                                     nb_rev_b0s=int(n_rev > 0))

        index = create_index(bvals, n_rev=n_rev)

    bvecs = create_non_zero_norm_bvecs(bvecs)

    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    acqparams_path = os.path.join(args.out_directory, 'acqparams.txt')
    np.savetxt(acqparams_path, acqparams, fmt='%1.4f', delimiter=' ')
    index_path = os.path.join(args.out_directory, 'index.txt')
    np.savetxt(index_path, index, fmt='%i', newline=" ")
    bvecs_path = os.path.join(args.out_directory, 'non_zero_norm.bvecs')
    np.savetxt(bvecs_path, bvecs.T, fmt="%.8f")

    additional_args = ""
    if args.topup is not None:
        additional_args += "--topup={} ".format(args.topup)

    if args.slice_drop_correction:
        additional_args += "--repol "

    if args.fix_seed:
        additional_args += "--initrand "

    if args.lsr_resampling:
        if len(bvals) - n_rev == n_rev:
            forward_bb = bvals[:n_rev, None] * bvecs[:n_rev, :]
            reverse_bb = bvals[n_rev:, None] * bvecs[n_rev:, :]
            if np.allclose(forward_bb, reverse_bb):
                additional_args += "--resamp=lsr --fep=true "
            else:
                logging.warning('Least-square resampling disabled since '
                                'directions in both phase directions differ.')
        else:
            logging.warning('Least-square resampling disabled since number of '
                            'directions in both phase directions differ.')

    if args.eddy_options:
        additional_args += args.eddy_options

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
