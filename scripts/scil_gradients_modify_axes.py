#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flip (ex, x --> -x) or swap (ex, x <-> y) chosen axes of the gradient sampling
matrix. Result will be saved in the same format as input gradient sampling
file.

Formerly: scil_flip_gradients.py or scil_swap_gradient_axis.py
"""
import argparse
import logging
import os

import numpy as np

from scilpy.gradients.bvec_bval_tools import (flip_gradient_sampling,
                                              swap_gradient_axis)
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_gradient_sampling_file',
                   help='Path to gradient sampling file. (.bvec or .b)')

    p.add_argument('out_gradient_sampling_file',
                   help='Where to save the flipped gradient sampling file.'
                        'Extension (.bvec or .b) must be the same as '
                        'in_gradient_sampling_file')

    # Note: We can't ask for text such as '-x' or '-xyz' because -x is
    # understood as another option -x by the argparser. With ints, we can
    # add a negative value. Not using 0, 1, 2 so that we can use the
    # negative value (-0 would not work)
    p.add_argument('final_order', type=int, nargs=3,
                   choices=[1, 2, 3, -1, -2, -3],
                   help="The final order of the axes, compared to original "
                        "order: x=1 y=2 z=3.\n"
                        "Ex: to only flip y: 1 -2 3.\n"
                        "Ex: to only swap x and y: 2 1 3.\n"
                        "Ex: to first flip x, then permute all three axes: "
                        " 3 -1 2.")

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_gradient_sampling_file)
    assert_outputs_exist(parser, args, args.out_gradient_sampling_file)

    _, ext_in = os.path.splitext(args.in_gradient_sampling_file)
    if ext_in not in ['.bvec', '.b']:
        parser.error('Extension for MRtrix format should .b, and extension '
                     'for FSL format should be .bvec. Got {}, we do not know '
                     'how to interpret.'.format(ext_in))
    _, ext_out = os.path.splitext(args.out_gradient_sampling_file)
    if ext_in != ext_out:
        parser.error("Output format (.bvec or .b) should be the same as the "
                     "input's. We do not support conversion in this script.")

    # Format final order
    # Our scripts use axes as 0, 1, 2 rather than 1, 2, 3: adding -1.
    axes_to_flip = []
    swapped_order = []
    for next_axis in args.final_order:
        if next_axis in [1, 2, 3]:
            swapped_order.append(next_axis - 1)
        elif next_axis in [-1, -2, -3]:
            axes_to_flip.append(abs(next_axis) - 1)
            swapped_order.append(abs(next_axis) - 1)
        else:
            parser.error("Sorry, final order not understood.")

    # Verifying that user did not ask for, ex, -xxy
    if len(np.unique(swapped_order)) != 3:
        parser.error("final_order should contain the three axis.")

    # Could use io.gradients method, but we don't have a bvals file here, only
    # treating with the bvecs. Loading directly.
    bvecs = np.loadtxt(args.in_gradient_sampling_file)
    if ext_in == '.bvec':
        # Supposing FSL format: Per columns
        if bvecs.shape[0] != 3:
            parser.error("b-vectors format for a .b file should be FSL, "
                         "and contain 3 lines (x, y, z), but got {}"
                         .format(bvecs.shape[0]))
        bvecs = flip_gradient_sampling(bvecs, axes_to_flip, 'fsl')
        bvecs = swap_gradient_axis(bvecs, swapped_order, 'fsl')
        np.savetxt(args.out_gradient_sampling_file, bvecs, "%.8f")
    else:  # ext == '.b':
        # Supposing mrtrix format
        if bvecs.shape[1] != 4:
            parser.error("b-vectors format for a .b file should be mrtrix, "
                         "and contain 4 columns (x, y, z, bval), but got {}"
                         .format(bvecs.shape[1]))
        bvecs = flip_gradient_sampling(bvecs, axes_to_flip, 'mrtrix')
        bvecs = swap_gradient_axis(bvecs, swapped_order, 'mrtrix')
        np.savetxt(args.out_gradient_sampling_file, bvecs,
                   "%.8f %.8f %.8f %0.6f")


if __name__ == "__main__":
    main()
