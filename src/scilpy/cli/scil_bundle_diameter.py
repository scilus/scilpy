#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to estimate the diameter of bundle(s) along their length.
See also scil_bundle_shape_measures.py, which prints a quick estimate of
the diameter (volume / length). The computation here is more complex and done
for each section of the bundle.

The script expects:
- bundles with coherent endpoints from scil_tractogram_uniformize_endpoints.py
- labels maps of the bundle's sections, with around 5-50 points,
  from scil_bundle_label_map.py
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

Formerly: scil_estimate_bundles_diameter.py
"""

import argparse
import json
import logging
import os

from fury import actor
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_headers_compatible,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             parser_color_type)
from scilpy.tractanalysis.bundle_operations import project_to_cross_section
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
                   help='List of labels maps that match the bundles.')

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


def snapshot(scene, win_dims, output_filename):
    # Legacy. When this snapshotting gets updated to align with the
    # viz module, snapshot_scenes should be called directly
    snapshot = next(snapshot_scenes([scene], win_dims))
    img = compose_image(snapshot, win_dims, "NONE")
    img.save(output_filename)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

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

    stats = {}
    num_digits_labels = 3
    actor_list = []
    spatial_shape = nib.load(args.in_labels[0]).shape[:3]
    for i, filename in enumerate(args.in_bundles):
        sft = load_tractogram_with_reference(parser, args, filename)
        sft.to_vox()
        sft.to_corner()
        img_labels = nib.load(args.in_labels[i])

        data_labels = img_labels.get_fdata()
        bundle_name, _ = os.path.splitext(os.path.basename(filename))
        unique_labels = np.unique(data_labels)[1:].astype(int)

        # Empty bundle should at least return a json
        if not len(sft):
            tmp_dict = {}
            for label in unique_labels:
                tmp_dict['{}'.format(label).zfill(num_digits_labels)] \
                    = {'mean': 0.0, 'std': 0.0}
            stats[bundle_name] = {'diameter': tmp_dict}
            continue

        counter = 0
        labels_dict = {label: ([], []) for label in unique_labels}
        pts_labels = map_coordinates(data_labels,
                                     sft.streamlines._data.T-0.5,
                                     order=0, mode='nearest')
        # For each label, all positions and directions are needed to get
        # a tube estimation per label.
        for streamline in sft.streamlines:
            direction = np.gradient(streamline, axis=0).tolist()
            curr_labels = pts_labels[counter:counter+len(streamline)].tolist()

            for i, label in enumerate(curr_labels):
                if label > 0:
                    labels_dict[label][0].append(streamline[i])
                    labels_dict[label][1].append(direction[i])

            counter += len(streamline)

        centroid = np.zeros((len(unique_labels), 3))
        radius = np.zeros((len(unique_labels), 1))
        error = np.zeros((len(unique_labels), 1))
        for key in unique_labels:
            key = int(key)
            c, d, e = project_to_cross_section(labels_dict[key][0],
                                               labels_dict[key][1],
                                               args.fitting_func)
            centroid[key-1], radius[key-1], error[key-1] = c, d, e

        # Spatial smoothing to avoid degenerate estimation
        centroid_smooth = gaussian_filter(centroid, sigma=[1, 0],
                                          mode='nearest')
        centroid_smooth[::len(centroid)-1] = centroid[::len(centroid)-1]
        radius = gaussian_filter(radius, sigma=1, mode='nearest')
        error = gaussian_filter(error, sigma=1, mode='nearest')

        tmp_dict = {}
        for label in unique_labels:
            tmp_dict['{}'.format(label).zfill(num_digits_labels)] \
                = {'mean': float(radius[label-1][0])*2,
                   'std': float(error[label-1][0])}
        stats[bundle_name] = {'diameter': tmp_dict}

        if args.show_rendering or args.save_rendering:
            tube_actor = create_tube_with_radii(
                centroid_smooth, radius, error,
                wireframe=args.wireframe,
                error_coloring=args.error_coloring)
            actor_list.append(tube_actor)
            # TODO : move streamline actor to fury backend
            cmap = get_lookup_table('jet')
            coloring = cmap(pts_labels / np.max(pts_labels))[:, 0:3]
            streamlines_actor = actor.streamtube(sft.streamlines,
                                                 linewidth=args.width,
                                                 opacity=args.opacity,
                                                 colors=coloring)
            actor_list.append(streamlines_actor)

            slice_actor = actor.slicer(data_labels, np.eye(4))
            slice_actor.opacity(0.0)
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
        snapshot(scene, args.win_dims,
                 os.path.join(args.save_rendering, 'superior.png'))

        scene.pitch(180)
        scene.reset_camera()
        snapshot(scene, args.win_dims,
                 os.path.join(args.save_rendering, 'inferior.png'))

        scene.pitch(90)
        scene.set_camera(view_up=(0, 0, 1))
        scene.reset_camera()
        snapshot(scene, args.win_dims,
                 os.path.join(args.save_rendering, 'posterior.png'))

        scene.pitch(180)
        scene.set_camera(view_up=(0, 0, 1))
        scene.reset_camera()
        snapshot(scene, args.win_dims,
                 os.path.join(args.save_rendering, 'anterior.png'))

        scene.yaw(90)
        scene.reset_camera()
        snapshot(scene, args.win_dims,
                 os.path.join(args.save_rendering, 'right.png'))

        scene.yaw(180)
        scene.reset_camera()
        snapshot(scene, args.win_dims,
                 os.path.join(args.save_rendering, 'left.png'))

    print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
