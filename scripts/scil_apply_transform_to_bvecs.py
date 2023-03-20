#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transform bvecs using an affine/rigid transformation.
"""

import argparse

import numpy as np

from dipy.io.gradients import read_bvals_bvecs

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, load_matrix_in_any_format)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.image import transform_anatomy


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_bvecs',
                   help='Path of the bvec file, in FSL format')
    p.add_argument('in_transfo',
                   help='Path of the file containing the 4x4 \n'
                        'transformation, matrix (.txt, .npy or .mat).')
    p.add_argument('out_bvecs',
                   help='Output filename of the transformed bvecs.')

    p.add_argument('--inverse', action='store_true',
                   help='Apply the inverse transformation.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_bvecs, args.in_transfo])
    assert_outputs_exist(parser, args, args.out_bvecs)

    transfo = load_matrix_in_any_format(args.in_transfo)[:3, :3]

    if args.inverse:
        transfo = np.linalg.inv(transfo)

    _, bvecs = read_bvals_bvecs(None, args.in_bvecs)

    bvecs = bvecs @ transfo

    np.savetxt(str(args.out_bvecs), bvecs.T)


if __name__ == "__main__":
    main()
