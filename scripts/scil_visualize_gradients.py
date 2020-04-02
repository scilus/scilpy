#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vizualisation for sampling schemes.
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
from scilpy.viz.sampling_scheme import (build_ms_from_shell_idx,
                                        plot_each_shell,
                                        plot_proj_shell)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    p.add_argument(
        'scheme_file', metavar='scheme_file', nargs='+',
        help='Sampling scheme filename. (only accepts .bvecs and .bvals '
             'or .b).')

    p.add_argument(
        '--dis-sym', action='store_false', dest='enable_sym',
        help='Disable antipodal symmetry.')
    p.add_argument(
        '--out',
        help='Output file name picture without extension ' +
             '(will be png file(s)).')
    p.add_argument(
        '--res', type=int, default=(300, 300), nargs='+',
        help='Resolution of the output picture(s).')

    g1 = p.add_argument_group(title='Enable/Disable renderings.')
    g1.add_argument(
        '--dis-sphere', action='store_false', dest='enable_sph',
        help='Disable the rendering of the sphere.')
    g1.add_argument(
        '--dis-proj', action='store_false', dest='enable_proj',
        help='Disable rendering of the projection supershell.')
    g1.add_argument(
        '--plot-shells', action='store_true', dest='plot_shells',
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
    assert_inputs_exist(parser, args.scheme_file)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if len(args.scheme_file) == 2:
        assert_gradients_filenames_valid(parser, args.scheme_file, 'fsl')
    elif len(args.scheme_file) == 1:
        assert_gradients_filenames_valid(parser, args.scheme_file, 'mrtrix')
    else:
        parser.error('Depending on the gradient format you should have '
                     'two files for FSL format and one file for MRtrix')

    out_basename = None
    if args.out:
        out_basename, ext = os.path.splitext(args.out)
        possibleOutputPaths = [out_basename + '_shell_' + str(i) +
                               '.png' for i in range(30)]
        possibleOutputPaths.append(out_basename + '.png')
        assert_outputs_exist(parser, args, possibleOutputPaths)

    proj = args.enable_proj
    each = args.plot_shells

    if not (proj or each):
        parser.error('Select at least one type of rendering (proj or each).')

    if len(args.scheme_file) == 2:
        scheme_files = args.scheme_file
        scheme_files.sort()  # [bval, bvec]
        # bvecs/bvals (FSL) format, X Y Z AND b (or transpose)
        points = np.genfromtxt(scheme_files[1])
        if points.shape[0] == 3:
            points = points.T
        bvals = np.genfromtxt(scheme_files[0])
        centroids, shell_idx = identify_shells(bvals)
    else:
        # MRtrix format X, Y, Z, b
        scheme_file = args.scheme_file[0]
        tmp = np.genfromtxt(scheme_file, delimiter=' ')
        points = tmp[:, :3]
        bvals = tmp[:, 3]
        centroids, shell_idx = identify_shells(bvals)

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
                        ofile=out_basename, ores=tuple(args.res))
    if each:
        plot_each_shell(ms, centroids, plot_sym_vecs=sym, use_sphere=sph, same_color=same,
                        rad=0.025, opacity=args.opacity,
                        ofile=out_basename, ores=tuple(args.res))


if __name__ == "__main__":
    main()
