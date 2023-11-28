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

from scilpy.gradients.bvec_bval_tools import (flip_gradient_sampling,
                                              swap_gradient_axis,
                                              str_to_axis_index)
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_gradient_sampling_file',
                   help='Path to gradient sampling file. (.bvec or .b)')

    p.add_argument('out_gradient_sampling_file',
                   help='Where to save the flipped gradient sampling file.')

    # Note: We can't ask for 3 separate nargs because -x is understood as
    # another option -x.
    p.add_argument('final_order',
                   help="The final order of the axis, compared to original "
                        "order. \n"
                        "Choices: ['x', 'y', 'z', '-x', '-y', '-z']\n"
                        "Ex: to only flip y: --final_order x-yz.\n"
                        "Ex: to only swap x and y: --final_order yxz.\n"
                        "Ex: to first flip x, then permute all three axes: "
                        "--final_order z-xy.")

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_gradient_sampling_file)
    assert_outputs_exist(parser, args, args.out_gradient_sampling_file)

    _, ext = os.path.splitext(args.in_gradient_sampling_file)
    if ext not in ['.bvec', '.b']:
        parser.error('Extension for MRtrix format should .b, and extension '
                     'for FSL format should be .bvec. Got {}, we do not know '
                     'how to interpret.'.format(ext))

    # Format final order
    axes_to_flip = []
    swapped_order = []
    next_axis = ''
    for char in args.final_order:
        next_axis += char
        if char != '-':
            if next_axis in ['x', 'y', 'z']:
                swapped_order.append(str_to_axis_index(next_axis))
            elif next_axis in ['-x', '-y', '-z']:
                axes_to_flip.append(str_to_axis_index(next_axis[1]))
                swapped_order.append(str_to_axis_index(next_axis[1]))
            else:
                parser.error("Sorry, final_order not understood.")
            next_axis = ''

    # Verifying that user did not ask for, ex, -xxy
    if len(np.unique(swapped_order)) != 3:
        parser.error("final_order should contain the three axis.")

    bvecs = np.loadtxt(args.in_gradient_sampling_file)
    if ext == '.bvec':
        # Supposing FSL format
        bvecs = flip_gradient_sampling(bvecs, axes_to_flip, 'fsl')
        bvecs = swap_gradient_axis(bvecs, swapped_order, 'fsl')
        np.savetxt(args.out_gradient_sampling_file, bvecs, "%.8f")
    else:  # ext == '.b':
        # Supposing mrtrix format
        bvecs = flip_gradient_sampling(bvecs, axes_to_flip, 'mrtrix')
        bvecs = swap_gradient_axis(bvecs, swapped_order, 'mrtrix')
        np.savetxt(args.out_gradient_sampling_file, bvecs,
                   "%.8f %.8f %.8f %0.6f")


if __name__ == "__main__":
    main()
