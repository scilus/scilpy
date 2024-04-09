#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vizualisation for directions on a sphere, either from a gradient sampling (i.e.
a list of b-vectors) or from a Dipy sphere.
"""

import argparse
import logging
import numpy as np
import os

from dipy.data import get_sphere

from scilpy.gradients.bvec_bval_tools import identify_shells
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_gradients_filenames_valid,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.viz.gradients import (build_ms_from_shell_idx,
                                  plot_each_shell,
                                  plot_proj_shell)

sphere_choices = ['symmetric362', 'symmetric642', 'symmetric724',
                  'repulsion724', 'repulsion100', 'repulsion200']


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        '--in_gradient_scheme', nargs='+',
        help='Gradient sampling filename. (only accepts .bvec and .bval '
             'together or only .b).')
    g.add_argument(
        '--dipy_sphere', choices=sphere_choices,
        help="Dipy sphere choice.")

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

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # -- Perform checks
    if args.in_gradient_scheme is not None:
        assert_inputs_exist(parser, args.in_gradient_scheme)

        if len(args.in_gradient_scheme) == 2:
            assert_gradients_filenames_valid(parser, args.in_gradient_scheme,
                                             'fsl')
        elif len(args.in_gradient_scheme) == 1:
            basename, ext = os.path.splitext(args.in_gradient_scheme[0])
            if ext in ['.bvec', '.bvecs', '.bvals', '.bval']:
                parser.error('You should input two files for fsl format '
                             '(.bvec and .bval).')
            else:
                assert_gradients_filenames_valid(parser,
                                                 args.in_gradient_scheme,
                                                 'mrtrix')
        else:
            parser.error('Depending on the gradient format you should have '
                         'two files for FSL format and one file for MRtrix')

    out_basename = None

    proj = args.enable_proj
    each = args.plot_shells

    if not (proj or each):
        parser.error('Select at least one type of rendering '
                     '(dis-proj or plot_shells).')
    if args.dipy_sphere is not None:
        # Only one sphere. Ignoring proj option.
        proj = True
        each = False

    # -- Ok. Now prepare vertices to show
    if args.in_gradient_scheme is not None:
        if len(args.in_gradient_scheme) == 2:
            in_gradient_schemes = args.in_gradient_scheme
            in_gradient_schemes.sort()  # [bval, bvec]
            # bvecs/bvals (FSL) format, X Y Z AND b (or transpose)
            points = np.genfromtxt(in_gradient_schemes[1])
            if points.shape[0] == 3:
                points = points.T
            bvals = np.genfromtxt(in_gradient_schemes[0])
            centroids, shell_idx = identify_shells(bvals)
        else:
            # MRtrix format X, Y, Z, b
            in_gradient_scheme = args.in_gradient_scheme[0]
            tmp = np.genfromtxt(in_gradient_scheme, delimiter=' ')
            points = tmp[:, :3]
            bvals = tmp[:, 3]
            centroids, shell_idx = identify_shells(bvals)

        if args.verbose:
            logging.info("Found {} centroids: {}".format(
                len(centroids), centroids))

        if args.out_basename:
            out_basename, ext = os.path.splitext(args.out_basename)
            possible_output_paths = [out_basename + '_shell_' + str(i) +
                                     '.png' for i in centroids]
            possible_output_paths.append(out_basename + '.png')
            assert_outputs_exist(parser, args, possible_output_paths)

        indexes = []
        for idx in np.where(centroids < 40)[0]:
            if args.verbose:
                logging.info("Removing bval = {} "
                             "from display".format(centroids[idx]))

            indexes.append(idx)
            shell_idx[shell_idx == idx] = -1
            centroids = np.delete(centroids,
                                  np.where(centroids == centroids[idx]))

        indexes = np.asarray(indexes)
        if len(shell_idx[shell_idx == -1]) > 0:
            for idx, val in enumerate(shell_idx):
                if val != 0 and val != -1:
                    shell_idx[idx] -= len(np.where(indexes < val)[0])
        ms = build_ms_from_shell_idx(points, shell_idx)

    else:
        ms = [get_sphere(args.dipy_sphere).vertices]
        centroids = None  # plot_each_shell not used.

    sym = args.enable_sym
    sph = args.enable_sph
    same = args.same_color

    if proj:
        plot_proj_shell(ms, use_sym=sym, use_sphere=sph, same_color=same,
                        rad=0.025, opacity=args.opacity,
                        ofile=out_basename, ores=(args.res, args.res))
    if each:
        plot_each_shell(ms, centroids, plot_sym_vecs=sym, use_sphere=sph,
                        same_color=same, rad=0.025, opacity=args.opacity,
                        ofile=out_basename, ores=(args.res, args.res))


if __name__ == "__main__":
    main()
