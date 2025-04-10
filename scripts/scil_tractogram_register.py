#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a linear transformation matrix from the registration of 2 tractograms.
Typically, this script is run before scil_tractogram_apply_transform.py.

For more information on how to use the various registration scripts, see the
doc at:
https://scilpy.readthedocs.io/en/latest/documentation/tractogram_registration.html

Formerly: scil_register_tractogram.py
----------------------------------------------------------------------------------
Reference:
[1] E. Garyfallidis, O. Ocegueda, D. Wassermann, M. Descoteaux
    Robust and efficient linear registration of white-matter fascicles in the
    space of streamlines, NeuroImage, Volume 117, 15 August 2015, Pages 124-140
    (http://www.sciencedirect.com/science/article/pii/S1053811915003961)
----------------------------------------------------------------------------------
"""

import argparse
import logging
import os

from dipy.align.streamlinear import whole_brain_slr
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('moving_tractogram',
                   help='Path of the moving tractogram.')
    p.add_argument('static_tractogram',
                   help='Path of the target tractogram.')

    p.add_argument('--out_name', default='transformation.txt',
                   help='Filename of the transformation matrix. \n'
                        'The registration type will be appended as a suffix,\n'
                        '[<out_name>_<affine/rigid>.txt]. \n'
                        'Default: [%(default)s]')
    p.add_argument('--only_rigid', action='store_true',
                   help='If set, will only use a rigid transformation '
                        '(uses affine by default).')

    add_reference_arg(p, 'moving_tractogram')
    add_reference_arg(p, 'static_tractogram')
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.moving_tractogram,
                                 args.static_tractogram],
                        [args.moving_tractogram_ref,
                         args.static_tractogram_ref])

    if args.only_rigid:
        matrix_filename = os.path.splitext(args.out_name)[0] + '_rigid.txt'
    else:
        matrix_filename = os.path.splitext(args.out_name)[0] + '_affine.txt'

    assert_outputs_exist(parser, args, matrix_filename, args.out_name)

    sft_moving = load_tractogram_with_reference(parser, args,
                                                args.moving_tractogram,
                                                arg_name='moving_tractogram')

    sft_static = load_tractogram_with_reference(parser, args,
                                                args.static_tractogram,
                                                arg_name='static_tractogram')

    if args.only_rigid:
        transformation_type = 'rigid'
    else:
        transformation_type = 'affine'

    ret = whole_brain_slr(sft_moving.streamlines,
                          sft_static.streamlines,
                          x0=transformation_type,
                          maxiter=150, greater_than=1,
                          verbose=args.verbose)
    _, transfo, _, _ = ret

    np.savetxt(matrix_filename, transfo)


if __name__ == "__main__":
    main()
