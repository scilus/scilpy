#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to smooth a surface with a Laplacian blur.

step_size from 0.1 to 10 is recommended
Smoothing_time = step_size * nb_steps
    [1, 10] for a small smoothing
    [10, 100] for a moderate smoothing
    [100, 1000] for a big smoothing
"""

import argparse
import logging

import numpy as np
from trimeshpy.io import load_mesh_from_file

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_surface',
                   help='Input surface (.vtk).')

    p.add_argument('out_surface',
                   help='Output smoothed surface (.vtk).')

    p.add_argument('-m', '--vts_mask',
                   help='Vertices mask, where to apply the flow (.npy).')

    p.add_argument('-n', '--nb_steps', type=int, default=2,
                   help='Number of steps for laplacian smooth [%(default)s].')

    p.add_argument('-s', '--step_size', type=float, default=5.0,
                   help='Laplacian smooth step size [%(default)s].')

    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_surface, args.vts_mask)
    assert_outputs_exist(parser, args, args.out_surface)

    # Check smoothing parameters
    if args.nb_steps < 1:
        parser.error("Number of steps should be strictly positive")

    if args.step_size <= 0.0:
        parser.error("Step size should be strictly positive")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Step size (zero for masked vertices)
    if args.vts_mask:
        mask = np.load(args.vts_mask)
        step_size_per_vts = args.step_size * mask.astype(float)
    else:
        step_size_per_vts = args.step_size

    mesh = load_mesh_from_file(args.in_surface)

    # Laplacian smoothing
    vts = mesh.laplacian_smooth(
        nb_iter=args.nb_steps,
        diffusion_step=step_size_per_vts,
        backward_step=True)
    mesh.set_vertices(vts)

    # Save Mesh Surface
    mesh.save(args.out_surface)


if __name__ == "__main__":
    main()
