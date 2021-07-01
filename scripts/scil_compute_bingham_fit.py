#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import argparse

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist
from scilpy.reconst.bingham import bingham_fit_sh_volume


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('in_sh')
    p.add_argument('out_bingham')

    p.add_argument('--max_lobes', type=int, default=5)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_sh)
    assert_outputs_exist(parser, args, args.out_bingham)

    sh_im = nib.load(args.in_sh)
    data = sh_im.get_fdata()
    out = bingham_fit_sh_volume(data, args.max_lobes)

    nib.save(nib.Nifti1Image(out, sh_im.affine), args.out_bingham)


if __name__ == '__main__':
    main()
