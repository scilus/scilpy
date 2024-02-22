#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a typical command for topup and create the necessary files.
The reversed b0 must be in a different file.

Formerly: scil_prepare_topup_command.py
"""

import argparse
import logging
import os
import subprocess

import nibabel as nib
import numpy as np
from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             assert_fsl_options_exist)
from scilpy.preprocessing.distortion_correction import create_acqparams


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_forward_b0',
                   help='Input b0 Nifti image with forward phase encoding.')

    p.add_argument('in_reverse_b0',
                   help='Input b0 Nifti image with reversed phase encoding.')

    p.add_argument('--config', default='b02b0.cnf',
                   help='Topup config file [%(default)s].')

    p.add_argument('--synb0', action='store_true',
                   help='If set, will use SyNb0 custom acqparams file.')

    p.add_argument('--encoding_direction', default='y',
                   choices=['x', 'y', 'z'],
                   help='Acquisition direction of the forward b0 '
                        'image, default is AP [%(default)s].')

    p.add_argument('--readout', type=float, default=0.062,
                   help='Total readout time from the DICOM metadata '
                        '[%(default)s].')

    p.add_argument('--out_b0s', default='fused_b0s.nii.gz',
                   help='Output fused b0 file [%(default)s].')

    p.add_argument('--out_directory', default='.',
                   help='Output directory for topup files [%(default)s].')

    p.add_argument('--out_prefix', default='topup_results',
                   help='Prefix of the topup results [%(default)s].')

    p.add_argument('--out_params', default='acqparams.txt',
                   help='Filename for the acquisition '
                        'parameters file [%(default)s].')

    p.add_argument('--out_script', action='store_true',
                   help='If set, will output a .sh script (topup.sh).\n' +
                        'else, will output the lines to the ' +
                        'terminal [%(default)s].')

    p.add_argument('--topup_options',  default='',
                   help='Additional options you want to use to run topup.\n'
                        'Add these options using quotes (i.e. "--fwhm=6'
                        ' --miter=4").')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    try:
        devnull = open(os.devnull)
        subprocess.call("topup", stderr=devnull)
    except:
        logging.warning(
            "topup not found. If executing locally, please install "
            "the command from the FSL library and make sure it is "
            "available in your path.")

    required_args = [args.in_forward_b0, args.in_reverse_b0]

    assert_inputs_exist(parser, required_args)
    assert_outputs_exist(parser, args, [], args.out_b0s)
    assert_fsl_options_exist(parser, args.topup_options, 'topup')

    if os.path.splitext(args.out_prefix)[1] != '':
        parser.error('The prefix must not contain any extension.')

    b0_img = nib.load(args.in_forward_b0)
    b0 = b0_img.get_fdata(dtype=np.float32)

    if len(b0.shape) == 4 and b0.shape[3] > 1:
        logging.warning("B0 is 4D. To speed up Topup, we recommend "
                        "using only one b0 in both phase encoding "
                        "direction, unless necessary.")
    elif len(b0.shape) == 3:
        b0 = b0[..., None]

    rev_b0_img = nib.load(args.in_reverse_b0)
    rev_b0 = rev_b0_img.get_fdata(dtype=np.float32)

    if len(rev_b0.shape) == 4 and rev_b0.shape[3] > 1:
        logging.warning("Reverse B0 is 4D. To speed up Topup, we "
                        "recommend using only one b0 in both phase "
                        "encoding direction, unless necessary.")
    elif len(rev_b0.shape) == 3:
        rev_b0 = rev_b0[..., None]

    fused_b0s = np.concatenate((b0, rev_b0), axis=-1)
    fused_b0s_path = os.path.join(args.out_directory, args.out_b0s)
    nib.save(nib.Nifti1Image(fused_b0s, b0_img.affine), fused_b0s_path)

    acqparams = create_acqparams(args.readout, args.encoding_direction,
                                 args.synb0, b0.shape[-1], rev_b0.shape[-1])

    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    acqparams_path = os.path.join(args.out_directory, args.out_params)
    np.savetxt(acqparams_path, acqparams, fmt='%1.4f', delimiter=' ')

    output_path = os.path.join(args.out_directory, args.out_prefix)
    fout_path = os.path.join(args.out_directory, "correction_field")
    iout_path = os.path.join(args.out_directory, "corrected_b0s")

    additional_args = ""
    if args.topup_options:
        additional_args = args.topup_options

    topup = 'topup --imain={0} --datain={1}'\
            ' --config={2} --verbose --out={3}'\
            ' --fout={4} --iout={5} --subsamp=1 {6}\n'\
        .format(fused_b0s_path, acqparams_path, args.config, output_path,
                fout_path, iout_path, additional_args)

    if args.out_script:
        with open("topup.sh", 'w') as f:
            f.write(topup)
    else:
        print(topup)


if __name__ == '__main__':
    main()
