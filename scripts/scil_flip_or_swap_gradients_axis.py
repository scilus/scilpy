#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flip (ex, x --> -x) or swap (ex, x <-> y) chosen axes of the gradient sampling
matrix. Result will be saved in the same format as input gradient sampling
file.
"""
import argparse
import os

import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.gradients.bvec_bval_tools import (flip_gradient_sampling,
                                              flip_fsl_gradient_sampling, swap_gradient_axis,
                                              swap_mrtrix_gradient_axis)
from scilpy.utils.util import str_to_index


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_gradient_sampling_file',
                   help='Path to gradient sampling file. (.bvec or .b)')

    p.add_argument('out_gradient_sampling_file',
                   help='Where to save the flipped gradient sampling file.')

    p.add_argument('--final_order', metavar='dimension',
                   choices=['x', 'y', 'z', '-x', '-y', '-z'], nargs=3,
                   help='The final order of the axis, compared to original '
                        'order.\n'
                        'Ex: to only flip y: --final_order x -y z.\n'
                        'Ex: to only swap x and y: --final_order y x z.\n'
                        'Ex: to first flip x, then permute all three axes: '
                        '--final_order z -x y.')

    gradients_type = p.add_mutually_exclusive_group(required=True)
    gradients_type.add_argument('--fsl', dest='fsl_bvecs',
                                action='store_true',
                                help='Specify fsl format.')
    gradients_type.add_argument('--mrtrix', dest='fsl_bvecs',
                                action='store_false',
                                help='Specify mrtrix format.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_gradient_sampling_file)
    assert_outputs_exist(parser, args, args.out_gradient_sampling_file)

    # Separating into flip and swap
    axes_to_flip = []
    swapped_order = []
    for axis in args.final_order:
        if axis[0] == '-':
            axes_to_flip.append(axis[1])
            axis = axis[1]
        swapped_order.append(axis)
    axes_to_flip = [str_to_index(axis) for axis in axes_to_flip]
    swapped_order = [str_to_index(axis) for axis in swapped_order]

    # Verifying that user did not ask for, ex, -x x y
    if len(np.unique(swapped_order)) != 3:
        parser.error("--final_order should contain the three axis.")

    _, ext = os.path.splitext(args.in_gradient_sampling_file)
    bvecs = np.loadtxt(args.in_gradient_sampling_file)
    if args.fsl_bvecs:
        if ext == '.bvec':
            bvecs = flip_gradient_sampling(bvecs, axes_to_flip, 'fsl')
            bvecs = swap_gradient_axis(bvecs, swapped_order, 'fsl')
            np.savetxt(args.out_gradient_sampling_file, bvecs, "%.8f")
        else:
            parser.error('Extension for FSL format should be .bvec.')

    elif ext == '.b':
        bvecs = flip_gradient_sampling(bvecs, axes_to_flip, 'mrtrix')
        bvecs = swap_gradient_axis(bvecs, swapped_order, 'mrtrix')
        np.savetxt(args.out_gradient_sampling_file, bvecs,
                   "%.8f %.8f %.8f %0.6f")
    else:
        parser.error('Extension for MRtrix format should .b.')


if __name__ == "__main__":
    main()
