#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import logging
import os

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np
from scilpy.io.utils import add_overwrite_arg, \
    assert_inputs_exist, assert_outputs_exist
from scilpy.preprocessing.distortion_correction import create_acqparams

DESCRIPTION = """
Prepare a typical command for topup and create the necessary files.
The reversed b0 must be in a different file.
 """


def _build_arg_parser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input_dwi',
                   help='input DWI Nifti image')

    p.add_argument('bvals',
                   help='b-values file in FSL format')

    p.add_argument('bvecs',
                   help='b-vectors file in FSL format')

    p.add_argument('reverse_b0',
                   help='b0 image with reversed phase encoding.')

    p.add_argument('--config', default='b02b0.cnf',
                   help='topup config file [%(default)s].')

    p.add_argument('--b0_thr', type=float, default=20,
                   help='All b-values with values less than or equal ' +
                        'to b0_thr are considered as b0s i.e. without ' +
                        'diffusion weighting')

    p.add_argument('--encoding_direction', default='y', choices=['x', 'y', 'z'],
                   help='acquisition direction, default is AP-PA [%(default)s].')

    p.add_argument('--readout', type=float, default=0.062,
                   help='total readout time from the DICOM metadata [%(default)s].')

    p.add_argument('--output_b0s', default='fused_b0s.nii.gz',
                   help='output fused b0 file [%(default)s].')

    p.add_argument('--output_directory', default='.',
                   help='output directory for topup files [%(default)s].')

    p.add_argument('--output_prefix', default='topup_results',
                   help='prefix of the topup results [%(default)s].')

    p.add_argument('--output_script', action='store_true',
                   help='if set, will output a .sh script (topup.sh).\n' +
                        'else, will output the lines to the ' +
                        'terminal [%(default)s].')

    add_overwrite_arg(p)

    p.add_argument('--verbose', '-v', action='store_true',
                   help='produce verbose output.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    required_args = [args.input_dwi, args.bvals, args.bvecs, args.reverse_b0]

    assert_inputs_exist(parser, required_args)
    assert_outputs_exist(parser, args, [], args.output_b0s)

    if os.path.splitext(args.output_prefix)[1] != '':
        parser.error('The prefix must not contain any extension.')

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)
    bvals_min = bvals.min()

    # TODO refactor this
    b0_threshold = args.b0_thr
    if bvals_min < 0 or bvals_min > b0_threshold:
        raise ValueError('The minimal b-value is lesser than 0 or greater '
                         'than {0}. This is highly suspicious. Please check '
                         'your data to ensure everything is correct. '
                         'Value found: {1}'.format(b0_threshold, bvals_min))

    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)

    acqparams = create_acqparams(gtab, args.readout, args.encoding_direction)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    acqparams_path = os.path.join(args.output_directory, 'acqparams.txt')
    np.savetxt(acqparams_path, acqparams, fmt='%1.4f', delimiter=' ')

    rev_b0_img = nib.load(args.reverse_b0)
    rev_b0 = rev_b0_img.get_data()

    dwi_image = nib.load(args.input_dwi)
    dwi = dwi_image.get_data()
    b0s = dwi[..., gtab.b0s_mask]

    b0_idx = np.where(gtab.b0s_mask)[0]
    fused_b0s = np.zeros(b0s.shape[:-1]+(len(b0_idx)+1,))
    fused_b0s[..., 0:-1] = b0s
    fused_b0s[..., -1] = rev_b0
    fused_b0s_path = os.path.join(args.output_directory, args.output_b0s)
    nib.save(nib.Nifti1Image(fused_b0s,
                             rev_b0_img.affine),
             fused_b0s_path)

    output_path = os.path.join(args.output_directory, args.output_prefix)
    fout_path = os.path.join(args.output_directory, "correction_field")
    iout_path = os.path.join(args.output_directory, "corrected_b0s")
    topup = 'topup --imain={0} --datain={1}'\
            ' --config={2} --verbose --out={3}'\
            ' --fout={4} --iout={5} --subsamp=1 \n'\
        .format(fused_b0s_path, acqparams_path, args.config, output_path,
                fout_path, iout_path)

    if args.output_script:
        with open("topup.sh", 'w') as f:
            f.write(topup)
    else:
        print(topup)


if __name__ == '__main__':
    main()
