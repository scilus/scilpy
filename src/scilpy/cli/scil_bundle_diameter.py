#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to estimate the diameter of bundle(s) along their length.
See also scil_bundle_shape_measures, which prints a quick estimate of
the diameter (volume / length). The computation here is more complex and done
for each section of the bundle.

The script expects:
- bundles with coherent endpoints from scil_bundle_uniformize_endpoints
- labels maps of the bundle's sections, with around 5-50 points,
  from scil_bundle_label_map
    - <5 is not enough, high risk of bad fit
    - >50 is too much, high risk of bad fit
- bundles that are close to a tube
    - without major fanning, in a single axis
    - or if fanning is in 2 directions (uniform dispersion), leads to a good
      approximation

The scripts prints a JSON file with mean/std. This is compatible with our
tractometry scripts.
WARNING: STD is in fact an ERROR measure from the fit and NOT an STD.

Since the estimation and fit quality is not always intuitive for some bundles
and the tube with varying diameter is not easy to color/visualize,
the script comes with its own VTK rendering to allow exploration of the data.
(optional).
"""

import argparse
import json
import logging
import os

from fury import actor
import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_headers_compatible,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             parser_color_type)
from scilpy.tractanalysis.bundle_operations import compute_bundle_diameter
from scilpy.viz.backends.fury import (create_interactive_window,
                                      create_scene,
                                      snapshot_scenes)
from scilpy.viz.backends.vtk import create_tube_with_radii
from scilpy.viz.color import get_lookup_table
from scilpy.viz.screenshot import compose_image
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_bundles', nargs='+',
                   help='List of tractography files.')
    p.add_argument('in_labels', nargs='+',
                   help='List of labels maps that match the bundles, with '
                        'each section as a different label.')

    p.add_argument('--fitting_func', choices=['lin_up', 'lin_down', 'exp',
                                              'inv', 'log'],
                   help='Function to weigh points using their distance.\n'
                        '-  lin_up: linear\n'
                        '-  lin_down: inversely linear\n'
                        '-  exp: exponential\n'
                        '-  inv: inversely exponential\n'
                        '-  log: logarithmic\n'
                        '[Default: %(default)s]')

    p2 = p.add_argument_group(title='Visualization options')
    p3 = p2.add_mutually_exclusive_group()
    p3.add_argument('--show_rendering', action='store_true',
                    help='Display VTK window (optional).\n'
                         '(Note. This option is not verified by tests. If '
                         'you encounter any bug, \nplease report it to our '
                         'team.)')
    p3.add_argument('--save_rendering', metavar='OUT_FOLDER',
                    help='Save VTK render in the specified folder (optional)')
    p2.add_argument('--wireframe', action='store_true',
                    help='Use wireframe for the tube rendering.')
    p2.add_argument('--error_coloring', action='store_true',
                    help='Use the fitting error to color the tube.')
    p2.add_argument('--width', type=float, default=0.2,
                    help='Width of tubes or lines representing streamlines'
                    '\n[Default: %(default)s]')
    p2.add_argument('--opacity', type=float, default=0.2,
                    help='Opacity for the streamlines rendered with the tube.'
                    '\n[Default: %(default)s]')
    p2.add_argument("--win_dims", nargs=2, metavar=("WIDTH", "HEIGHT"),
                    default=(1920, 1080), type=int,
                    help="The dimensions for the vtk window. [%(default)s]")
    p2.add_argument('--background', metavar=('R', 'G', 'B'), nargs=3,
                    default=[1, 1, 1], type=parser_color_type,
                    help='RBG values [0, 255] of the color of the background.'
                    '\n[Default: %(default)s]')

    add_reference_arg(p)
    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _snapshot(scene, win_dims, output_filename):
    # Legacy. When this snapshotting gets updated to align with the
    # viz module, snapshot_scenes should be called directly
    snapshot = next(snapshot_scenes([scene], win_dims))
    img = compose_image(snapshot, win_dims, "NONE")
    img.save(output_filename)


def _prepare_bundle_view(args, sft, data_labels, centroid_smooth, radius,
                         error, pts_labels):
    tube_actor = create_tube_with_radii(
        centroid_smooth, radius, error,
        wireframe=args.wireframe,
        error_coloring=args.error_coloring)
    # TODO : move streamline actor to fury backend
    cmap = get_lookup_table('jet')
    coloring = cmap(pts_labels / np.max(pts_labels))[:, 0:3]
    streamlines_actor = actor.streamtube(sft.streamlines,
                                         linewidth=args.width,
                                         opacity=args.opacity,
                                         colors=coloring)

    slice_actor = actor.slicer(data_labels, np.eye(4))
    slice_actor.opacity(0.0)
    return tube_actor, streamlines_actor, slice_actor


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    ## Verifications

    # The number of labels maps must be equal to the number of bundles
    tmp = args.in_bundles + args.in_labels
    args.in_labels = args.in_bundles[(len(tmp) // 2):] + args.in_labels
    args.in_bundles = args.in_bundles[0:len(tmp) // 2]
    assert_inputs_exist(parser, args.in_bundles + args.in_labels,
                        args.reference)
    assert_output_dirs_exist_and_empty(parser, args, [],
                                       optional=args.save_rendering)

    # same subject: same header or coregistered subjects: same header
    assert_headers_compatible(parser, args.in_bundles + args.in_labels,
                              reference=args.reference)

    spatial_shape = nib.load(args.in_labels[0]).shape[:3]

    ## Processing
    # Most loading will be done inside the loop
    stats = {}
    actor_list = []
    for i, filename in enumerate(args.in_bundles):
        bundle_name, _ = os.path.splitext(os.path.basename(filename))
        logging.info("Computing data for bundle {}".format(filename))

        # Loading bundle and associated labels
        sft = load_tractogram_with_reference(parser, args, filename)
        sft.to_vox()
        sft.to_corner()
        img_labels = nib.load(args.in_labels[i])
        data_labels = img_labels.get_fdata()

        # Main computing
        (bundle_dict, centroid_smooth, radius, error,
         pts_labels) = compute_bundle_diameter(
            sft, data_labels, args.fitting_func)
        stats[bundle_name] = {'diameter': bundle_dict}

        if args.show_rendering or args.save_rendering:
            tube_actor, streamlines_actor, slice_actor = \
                _prepare_bundle_view(args, sft, data_labels, centroid_smooth,
                                     radius, error, pts_labels)
            actor_list.append(tube_actor)
            actor_list.append(streamlines_actor)
            actor_list.append(slice_actor)

    scene = create_scene(actor_list, "axial",
                         spatial_shape[2] // 2, spatial_shape,
                         args.win_dims[0] / args.win_dims[1],
                         bg_color=tuple(map(int, args.background)))

    # If there's actually streamlines to display
    if args.show_rendering:
        create_interactive_window(scene, args.win_dims, "image")
    elif args.save_rendering:
        # TODO : transform screenshotting to abide with viz module
        scene.reset_camera()
        _snapshot(scene, args.win_dims,
                  os.path.join(args.save_rendering, 'superior.png'))

        scene.pitch(180)
        scene.reset_camera()
        _snapshot(scene, args.win_dims,
                  os.path.join(args.save_rendering, 'inferior.png'))

        scene.pitch(90)
        scene.set_camera(view_up=(0, 0, 1))
        scene.reset_camera()
        _snapshot(scene, args.win_dims,
                  os.path.join(args.save_rendering, 'posterior.png'))

        scene.pitch(180)
        scene.set_camera(view_up=(0, 0, 1))
        scene.reset_camera()
        _snapshot(scene, args.win_dims,
                  os.path.join(args.save_rendering, 'anterior.png'))

        scene.yaw(90)
        scene.reset_camera()
        _snapshot(scene, args.win_dims,
                  os.path.join(args.save_rendering, 'right.png'))

        scene.yaw(180)
        scene.reset_camera()
        _snapshot(scene, args.win_dims,
                  os.path.join(args.save_rendering, 'left.png'))

    print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
