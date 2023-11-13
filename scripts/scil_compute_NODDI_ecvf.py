#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute NODDI [1] extracellular volume fraction (ECVF) map from intracellular
volume fraction (ICVF) map.
"""

import argparse
import os

import nibabel as nib
import numpy as np

from scilpy.io.utils import add_overwrite_arg, assert_inputs_exist

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

    p.add_argument('in_icvf',
                   help='ICVF map from NODDI script.')

    p.add_argument('--out_prefix',
                   help='Prefix used to save ECFV map.')
    p.add_argument('--out_dir',
                   help='Output directory for the ECVF map.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_icvf)

    if args.out_dir is None:
        args.out_dir = './'

    # Load ICVF image
    icvf_image = nib.load(args.in_icvf)
    icvf = icvf_image.get_fdata(dtype=np.float32)

    # Compute ECVF NODDI image
    ecvf = 1 - icvf

    # Mask ECVF backgound with ICVF map
    ecvf[np.where(icvf == 0)] = 0

    if args.out_prefix:
        out_name = args.out_prefix + '__FIT_ECVF.nii.gz'
    else:
        out_name = 'FIT_ECVF.nii.gz'

    nib.save(nib.Nifti1Image(ecvf.astype(np.float32),
                             icvf_image.affine),
             os.path.join(args.out_dir, out_name))


if __name__ == "__main__":
    main()
