#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vizualisation for gradient sampling.
Only supports .bvec/.bval and .b (MRtrix).
"""

import argparse
import logging
import numpy as np
import os

from scilpy.utils.bvec_bval_tools import identify_shells
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_gradients_filenames_valid,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.viz.gradient_sampling import (build_ms_from_shell_idx,
                                          plot_each_shell,
                                          plot_proj_shell)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    p.add_argument(
        'gradient_sampling_file', metavar='gradient_sampling_file', nargs='+',
        help='Gradient sampling filename. (only accepts .bvec and .bval '
             'together or only .b).')

    p.add_argument(
        '--dis-sym', action='store_false', dest='enable_sym',
        help='Disable antipodal symmetry.')
    p.add_argument(
        '--out_basename',
        help='Output file name picture without extension ' +
             '(will be png file(s)).')
    p.add_argument(
        '--res', type=int, default=300,
        help='Resolution of the output picture(s).')

    g1 = p.add_argument_group(title='Enable/Disable renderings.')
    g1.add_argument(
        '--dis-sphere', action='store_false', dest='enable_sph',
        help='Disable the rendering of the sphere.')
    g1.add_argument(
        '--dis-proj', action='store_false', dest='enable_proj',
        help='Disable rendering of the projection supershell.')
    g1.add_argument(
        '--plot_shells', action='store_true',
        help='Enable rendering each shell individually.')

    g2 = p.add_argument_group(title='Rendering options.')
    g2.add_argument(
        '--same-color', action='store_true', dest='same_color',
        help='Use same color for all shell.')
    g2.add_argument(
        '--opacity', type=float, default=1.0,
        help='Opacity for the shells.')

    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, args.gradient_sampling_file)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if len(args.gradient_sampling_file) == 2:
        assert_gradients_filenames_valid(parser, args.gradient_sampling_file, 'fsl')
    elif len(args.gradient_sampling_file) == 1:
        basename, ext = os.path.splitext(args.gradient_sampling_file[0])
        if ext in ['.bvec', '.bvecs', '.bvals', '.bval']:
            parser.error('You should input two files for fsl format (.bvec '
                         'and .bval).')
        else:
            assert_gradients_filenames_valid(parser, args.gradient_sampling_file, 'mrtrix')
    else:
        parser.error('Depending on the gradient format you should have '
                     'two files for FSL format and one file for MRtrix')

    out_basename = None

    proj = args.enable_proj
    each = args.plot_shells

    if not (proj or each):
        parser.error('Select at least one type of rendering (proj or each).')

    if len(args.gradient_sampling_file) == 2:
        gradient_sampling_files = args.gradient_sampling_file
        gradient_sampling_files.sort()  # [bval, bvec]
        # bvecs/bvals (FSL) format, X Y Z AND b (or transpose)
        points = np.genfromtxt(gradient_sampling_files[1])
        if points.shape[0] == 3:
            points = points.T
        bvals = np.genfromtxt(gradient_sampling_files[0])
        centroids, shell_idx = identify_shells(bvals)
    else:
        # MRtrix format X, Y, Z, b
        gradient_sampling_file = args.gradient_sampling_file[0]
        tmp = np.genfromtxt(gradient_sampling_file, delimiter=' ')
        points = tmp[:, :3]
        bvals = tmp[:, 3]
        centroids, shell_idx = identify_shells(bvals)

    if args.out_basename:
        out_basename, ext = os.path.splitext(args.out_basename)
        possible_output_paths = [out_basename + '_shell_' + str(i) +
                                 '.png' for i in centroids]
        possible_output_paths.append(out_basename + '.png')
        assert_outputs_exist(parser, args, possible_output_paths)

    for b0 in centroids[centroids < 40]:
        shell_idx[shell_idx == b0] = -1
        centroids = np.delete(centroids,  np.where(centroids == b0))

    shell_idx[shell_idx != -1] -= 1

    sym = args.enable_sym
    sph = args.enable_sph
    same = args.same_color

    ms = build_ms_from_shell_idx(points, shell_idx)
    if proj:
        plot_proj_shell(ms, use_sym=sym, use_sphere=sph, same_color=same,
                        rad=0.025, opacity=args.opacity,
                        ofile=out_basename, ores=(args.res, args.res))
    if each:
        plot_each_shell(ms, centroids, plot_sym_vecs=sym, use_sphere=sph, same_color=same,
                        rad=0.025, opacity=args.opacity,
                        ofile=out_basename, ores=(args.res, args.res))


if __name__ == "__main__":
    main()
