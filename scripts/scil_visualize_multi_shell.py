#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.viz.sampling_scheme import (build_ms_from_shell_idx,
                                        build_shell_idx_from_bval,
                                        plot_each_shell,
                                        plot_proj_shell)


DESCRIPTION = """
Vizualisation for sampling schemes.
Only supports .caru, .txt (Philips), .dir or .dvs (Siemens), .bvecs/.bvals
and .b (MRtrix).
"""


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=DESCRIPTION)

    p.add_argument(
        'scheme_file', action='store', metavar='scheme_file',
        help='Sampling scheme filename. (only accepts .txt or .caru or '
             '.bvecs and .bvals or .b or .dir or .dvs)')

    p.add_argument(
        '--dis-sym', action='store_false', dest='enable_sym',
        help='Disable antipodal symmetry.')
    p.add_argument(
        '--out',
        help='Output file name picture without extension ' +
             '(will be png file(s))')
    p.add_argument(
        '--res', type=int, default=(300, 300), nargs='+',
        help='Resolution of the output picture(s)')

    g1 = p.add_argument_group(title='Enable/Disable renderings')
    g1.add_argument(
        '--dis-sphere', action='store_false', dest='enable_sph',
        help='Disable the rendering of the sphere.')
    g1.add_argument(
        '--dis-proj', action='store_false', dest='enable_proj',
        help='Disable rendering of the projection supershell.')
    g1.add_argument(
        '--plot-shells', action='store_true', dest='plot_shells',
        help='Enable rendering each shell individually.')

    g2 = p.add_argument_group(title='Rendering options')
    g2.add_argument(
        '--same-color', action='store_true', dest='same_color',
        help='Use same color for all shell.')
    g2.add_argument(
        '--opacity', type=float, default=1.0,
        help='Opacity for the shells.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, args.scheme_file)

    if args.out:
        out_basename, ext = split_name_with_nii(args.out)
        possibleOutputPaths = [out_basename + '_shell_' + str(i) +
                               '.png' for i in range(30)]
        possibleOutputPaths.append(out_basename + '.png')
        assert_outputs_exist(parser, args, possibleOutputPaths)

    proj = args.enable_proj
    each = args.plot_shells

    if not (proj or each):
        parser.error('Select at least one type of rendering (proj or each).')

    # In no way robust, assume the input is from generate_sampling_scheme.py
    # For bvec(s)/bval(s)/FSL format, uses bad assumption for Transpose
    scheme_file = args.scheme_file
    basename, ext = split_name_with_nii(scheme_file)

    if ext == 'caru':
        # Caruyer format, X Y Z shell_idx
        tmp = np.genfromtxt(scheme_file)
        points = tmp[:, :3]
        shell_idx = tmp[:, 3]

    elif ext == 'txt':
        # Philips format, X Y Z b
        tmp = np.genfromtxt(scheme_file)
        points = tmp[:, :3]
        bvals = tmp[:, 3]
        shell_idx = build_shell_idx_from_bval(bvals, shell_th=50)

    elif ext == 'bvecs':
        # bvecs/bvals (FSL) format, X Y Z AND b (or transpose)
        points = np.genfromtxt(scheme_file)
        if points.shape[0] == 3:
            points = points.T
        bvals = np.genfromtxt(basename + '.bvals')
        shell_idx = build_shell_idx_from_bval(bvals, shell_th=50)

    elif ext == 'bvec':
        # bvecs/bvals (FSL) format, X Y Z AND b (or transpose)
        logging.info('Should rename .bvec/.bval to .bvecs/.bvals')
        points = np.genfromtxt(scheme_file)
        if points.shape[0] == 3:
            points = points.T
        bvals = np.genfromtxt(basename + '.bval')
        shell_idx = build_shell_idx_from_bval(bvals, shell_th=50)

    elif ext == 'bvals':
        # bvecs/bvals (FSL) format, X Y Z AND b (or transpose)
        bvals = np.genfromtxt(scheme_file)
        points = np.genfromtxt(basename + '.bvecs')
        if points.shape[0] == 3:
            points = points.T
        shell_idx = build_shell_idx_from_bval(bvals, shell_th=50)

    elif ext == 'bval':
        # bvecs/bvals (FSL) format, X Y Z AND b (or transpose)
        logging.info('Should rename .bvec/.bval to .bvecs/.bvals')
        bvals = np.genfromtxt(scheme_file)
        points = np.genfromtxt(basename + '.bvec')
        if points.shape[0] == 3:
            points = points.T
        shell_idx = build_shell_idx_from_bval(bvals, shell_th=50)

    elif ext == 'dir' or ext == 'dvs':
        vect = []
        # Siemens format, X, Y, Z
        with open(scheme_file) as f:
            for line in f:
                if 'vector[' in line.lower():
                    vect.append([float(f) for f in line.split('=')[1][2:-3].split(',')])
        vect = np.array(vect)

        norms = np.linalg.norm(vect, axis=1)
        # ugly work around for the division by b0 / replacing NaNs with 0.0
        old_settings = np.seterr(divide='ignore', invalid='ignore')
        points = vect / norms[:, None]
        np.seterr(**old_settings)
        points[np.isnan(points)] = 0.0
        points[np.isinf(points)] = 0.0

        fake_bmax = 3000.
        shell_idx = build_shell_idx_from_bval(fake_bmax * norms**2,
                                              shell_th=50)

    elif ext == "b":
        # MRtrix format X, Y, Z, b
        tmp = np.genfromtxt(scheme_file, delimiter=',')
        points = tmp[:, :3]
        bvals = tmp[:, 3]
        shell_idx = build_shell_idx_from_bval(bvals, shell_th=50)

    else:
        logging.error('Unknown format (Only supports .caru, .txt (Philips),' +
                      ' .bvecs/.bvals (FSL), .b (MRtrix), .dir or ' +
                      '.dvs (Siemens))')

    sym = args.enable_sym
    sph = args.enable_sph
    same = args.same_color

    ms = build_ms_from_shell_idx(points, shell_idx)

    if proj:
        plot_proj_shell(ms, use_sym=sym, use_sphere=sph, same_color=same,
                        rad=0.025, opacity=args.opacity,
                        ofile=out_basename, ores=tuple(args.res))
    if each:
        plot_each_shell(ms, use_sym=sym, use_sphere=sph, same_color=same,
                        rad=0.025, opacity=args.opacity,
                        ofile=out_basename, ores=tuple(args.res))


if __name__ == "__main__":
    main()
