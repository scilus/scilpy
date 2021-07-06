#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dipy.reconst.shm import sh_to_sf
import numpy as np
import nibabel as nib
import argparse

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist
from scilpy.reconst.bingham import bingham_fit_sh_volume


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('in_sh', help='Input SH image.')
    p.add_argument('out_bingham', help='Output bingham functions image.')

    p.add_argument('--max_lobes', type=int, default=5,
                   help='Maximum number of lobes per voxel'
                        ' to extract. [%(default)s]')
    p.add_argument('--a_th', type=float, default=0.0,
                   help='Absolute threshold for peaks'
                        ' extraction [%(default)s].')
    p.add_argument('--r_th', type=float, default=0.2,
                   help='Relative threshold for peaks'
                        ' extraction [%(default)s].')
    p.add_argument('--min_sep_angle', type=float, default=25,
                   help='Minimum separation angle between'
                        ' two peaks [%(default)s].')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_sh)
    assert_outputs_exist(parser, args, args.out_bingham)

    sh_im = nib.load(args.in_sh)
    data = sh_im.get_fdata()
    out = bingham_fit_sh_volume(data, args.max_lobes)

    nib.save(nib.Nifti1Image(out, np.eye(4)), args.out_bingham)


if __name__ == '__main__':
    main()
