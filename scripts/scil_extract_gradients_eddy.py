#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract gradient from eddy outputs
With full AP-PA eddy outputs a full bvec bval (2x nb of dirs and bval)
that doesnt fit with the output dwi (1x nb of dir)
"""

import argparse

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bvec',
                   help='In bvec file.')
    p.add_argument('in_bval',
                   help='In bval file.')
    p.add_argument('nb_dirs', type=int,
			       help='Number of directions per DWI.')
    p.add_argument('out_bvec',
                   help='Out bvec file.')
    p.add_argument('out_bval',
                   help='Out bval file.')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_bval, args.in_bval])
    assert_outputs_exist(parser, args, [args.out_bval, args.out_bvec])

    """
    IN BVEC
    """
    in_bvec = np.genfromtxt(args.in_bvec)
    in_bvec_split = np.hsplit(in_bvec, int(in_bvec.shape[1]/args.nb_dirs))
    if len(in_bvec_split)==2:
        out_bvec = np.mean( np.array([ in_bvec_split[0], in_bvec_split[1] ]), axis=0 )
    else:
        out_bvec = in_bvec_split[0]
    np.savetxt(args.out_bvec, out_bvec, '%.8f')

    """
    IN BVAL
    """
    in_bval = np.genfromtxt(args.in_bval)
    np.savetxt(args.out_bval, in_bval[0:args.nb_dirs], '%.0f')

if __name__ == '__main__':
    main()
