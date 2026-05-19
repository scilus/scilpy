#!/usr/bin/env python3
"""
Perform anatomically-constrainedtractography for a tracer experiment (e.g.
from Allen Mouse Brain Connectivity Atlas). The script takes as input a WM mask and
projection density map for a tracer injection coregistered to the same space as a
diffusion MRI SH volume.
"""
import argparse
import logging
from scilpy.io.utils import add_overwrite_arg, add_verbose_arg, add_sh_basis_args


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh', help='Path of the input SH volume.')
    p.add_argument('in_wm_mask', help='Path of the input WM mask.')
    p.add_argument('in_projection_map', help='Path of the input projection density map.')
    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_sh_basis_args(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    


if __name__ == "__main__":
    main()
