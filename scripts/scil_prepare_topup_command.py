#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a typical command for topup and create the necessary files.
The reversed b0 must be in a different file.
"""

import argparse
import logging
import os

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.preprocessing.distortion_correction import create_acqparams


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_dwi',
                   help='input DWI Nifti image')

    p.add_argument('in_bvals',
                   help='b-values file in FSL format')

    p.add_argument('in_bvecs',
                   help='b-vectors file in FSL format')

    p.add_argument('in_reverse_b0',
                   help='b0 image with reversed phase encoding.')

    p.add_argument('--config', default='b02b0.cnf',
                   help='topup config file [%(default)s].')

    p.add_argument('--b0_thr', type=float, default=20,
                   help='All b-values with values less than or equal ' +
                        'to b0_thr are considered as b0s i.e. without ' +
                        'diffusion weighting')

    p.add_argument('--encoding_direction', default='y',
                   choices=['x', 'y', 'z'],
                   help='acquisition direction, default is AP-PA '
                        '[%(default)s].')

    p.add_argument('--readout', type=float, default=0.062,
                   help='total readout time from the DICOM metadata '
                        '[%(default)s].')

    p.add_argument('--out_b0s', default='fused_b0s.nii.gz',
                   help='output fused b0 file [%(default)s].')

    p.add_argument('--out_directory', default='.',
                   help='output directory for topup files [%(default)s].')

    p.add_argument('--out_prefix', default='topup_results',
                   help='prefix of the topup results [%(default)s].')

    p.add_argument('--out_script', action='store_true',
                   help='if set, will output a .sh script (topup.sh).\n' +
                        'else, will output the lines to the ' +
                        'terminal [%(default)s].')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    required_args = [args.in_dwi, args.in_bvals, args.in_bvecs,
                     args.in_reverse_b0]

    assert_inputs_exist(parser, required_args)
    assert_outputs_exist(parser, args, [], args.out_b0s)

    if os.path.splitext(args.out_prefix)[1] != '':
        parser.error('The prefix must not contain any extension.')

    bvals, bvecs = read_bvals_bvecs(args.in_bvals, args.in_bvecs)
    bvals_min = bvals.min()

    # TODO refactor this
    b0_threshold = args.b0_thr
    if bvals_min < 0 or bvals_min > b0_threshold:
        raise ValueError('The minimal b-value is lesser than 0 or greater '
                         'than {0}. This is highly suspicious. Please check '
                         'your data to ensure everything is correct. '
                         'Value found: {1}'.format(b0_threshold, bvals_min))

    rev_b0_img = nib.load(args.in_reverse_b0)
    rev_b0 = rev_b0_img.get_fdata(dtype=np.float32)

    if len(rev_b0.shape) == 4:
        logging.warning("Reverse B0 is 4D. To speed up Topup, the mean of all "
                        "reverse B0 will be taken.")
        rev_b0 = np.mean(rev_b0, axis=3)

    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)

    dwi_image = nib.load(args.in_dwi)
    dwi = dwi_image.get_fdata(dtype=np.float32)
    b0 = dwi[..., gtab.b0s_mask]

    if b0.shape[3] > 1:
        logging.warning("More than one B0 was found. To speed up Topup, "
                        "the mean of all B0 will be taken.")
        b0 = np.mean(b0, axis=3)
    else:
        b0 = np.squeeze(b0, axis=3)

    fused_b0s = np.zeros(b0.shape+(2,))
    fused_b0s[..., 0] = b0
    fused_b0s[..., 1] = rev_b0
    fused_b0s_path = os.path.join(args.out_directory, args.out_b0s)
    nib.save(nib.Nifti1Image(fused_b0s,
                             rev_b0_img.affine),
             fused_b0s_path)

    acqparams = create_acqparams(args.readout, args.encoding_direction)

    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    acqparams_path = os.path.join(args.out_directory, 'acqparams.txt')
    np.savetxt(acqparams_path, acqparams, fmt='%1.4f', delimiter=' ')

    output_path = os.path.join(args.out_directory, args.out_prefix)
    fout_path = os.path.join(args.out_directory, "correction_field")
    iout_path = os.path.join(args.out_directory, "corrected_b0s")
    topup = 'topup --imain={0} --datain={1}'\
            ' --config={2} --verbose --out={3}'\
            ' --fout={4} --iout={5} --subsamp=1 \n'\
        .format(fused_b0s_path, acqparams_path, args.config, output_path,
                fout_path, iout_path)

    if args.out_script:
        with open("topup.sh", 'w') as f:
            f.write(topup)
    else:
        print(topup)


if __name__ == '__main__':
    main()
