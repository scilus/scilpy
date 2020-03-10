#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

from dipy.align.streamlinear import whole_brain_slr
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np


from scilpy.io.streamlines import ichunk, load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)

DESCRIPTION = """
Generate a linear transformation matrix from the registration of
2 tractograms. Typically, this script is run before
scil_apply_transform_to_tractogram.py.

For more informations on how to use the various registration scripts
see the doc/tractogram_registration.md readme file
"""

EPILOG = """
References:
[1] E. Garyfallidis, O. Ocegueda, D. Wassermann, M. Descoteaux
Robust and efficient linear registration of white-matter fascicles in the
space of streamlines, NeuroImage, Volume 117, 15 August 2015, Pages 124-140
(http://www.sciencedirect.com/science/article/pii/S1053811915003961)
"""


def register_tractogram(moving_tractogram, static_tractogram,
                        only_rigid, amount_to_load, matrix_filename,
                        verbose):

    amount_to_load = max(250000, amount_to_load)

    moving_streamlines = next(ichunk(moving_tractogram.streamlines,
                                     amount_to_load))

    static_streamlines = next(ichunk(static_tractogram.streamlines,
                                     amount_to_load))

    if only_rigid:
        transformation_type = 'rigid'
    else:
        transformation_type = 'affine'

    ret = whole_brain_slr(ArraySequence(static_streamlines),
                          ArraySequence(moving_streamlines),
                          x0=transformation_type,
                          maxiter=150,
                          verbose=verbose)
    _, transfo, _, _ = ret
    np.savetxt(matrix_filename, transfo)


def _build_args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION, epilog=EPILOG)

    p.add_argument('moving_tractogram',
                   help='Path of the moving tractogram.')

    p.add_argument('static_tractogram',
                   help='Path of the target tractogram.')

    add_reference_arg(p, 'moving_tractogram')

    add_reference_arg(p, 'static_tractogram')

    p.add_argument('--out_name',
                   default='transformation.npy',
                   help='Filename of the transformation matrix, \n'
                        'the registration type will be appended as a suffix,\n'
                        '[<out_name>_<affine/rigid>.npy]')

    p.add_argument('--only_rigid', action='store_true',
                   help='Will only use a rigid transformation, '
                        'uses affine by default.')

    p.add_argument('--amount_to_load', type=int,
                   default=250000,
                   help='Amount of streamlines to load for each tractogram \n'
                        'using lazy load. [%(default)s]')

    add_overwrite_arg(p)

    add_verbose_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.moving_tractogram,
                                 args.static_tractogram])

    if args.only_rigid:
        matrix_filename = os.path.splitext(args.out_name)[0] + '_rigid.npy'
    else:
        matrix_filename = os.path.splitext(args.out_name)[0] + '_affine.npy'

    assert_outputs_exist(parser, args, matrix_filename, args.out_name)

    sft_moving = load_tractogram_with_reference(parser,
                                                args,
                                                args.moving_tractogram,
                                                bbox_check=True,
                                                arg_name='moving_tractogram')

    sft_static = load_tractogram_with_reference(parser,
                                                args,
                                                args.static_tractogram,
                                                bbox_check=True,
                                                arg_name='static_tractogram')

    register_tractogram(sft_moving, sft_static,
                        args.only_rigid, args.amount_to_load, matrix_filename,
                        args.verbose)


if __name__ == "__main__":
    main()
