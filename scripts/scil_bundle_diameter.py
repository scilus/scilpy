#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to estimate the diameter of bundle(s) along their length.
See also scil_bundle_shape_measures.py, which prints a quick estimate of
the diameter (volume / length). The computation here is more complex and done
for each section of the bundle.

The script expects:
- bundles with coherent endpoints from scil_uniformize_streamlines_endpoints.py
- labels maps with around 5-50 points scil_compute_bundle_voxel_label_map.py
    <5 is not enough, high risk of bad fit
    >50 is too much, high risk of bad fit
- bundles that are close to a tube
    without major fanning in a single axis
    fanning is in 2 directions (uniform dispersion) good approximation

The scripts prints a JSON file with mean/std to be compatible with tractometry.
WARNING: STD is in fact an ERROR measure from the fit and NOT an STD.

Since the estimation and fit quality is not always intuitive for some bundles
and the tube with varying diameter is not easy to color/visualize,
the script comes with its own VTK rendering to allow exploration of the data.
(optional).
"""

import argparse
import json
import os

from dipy.io.utils import is_header_compatible
from fury import window, actor
import nibabel as nib
import numpy as np
from scipy.linalg import svd
from scipy.ndimage import map_coordinates, gaussian_filter

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_json_args,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             parser_color_type,
                             snapshot)
from scilpy.viz.scene_utils import create_tube_with_radii
from scilpy.viz.utils import get_colormap


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundles', nargs='+',
                   help='List of tractography files.')
    p.add_argument('in_labels', nargs='+',
                   help='List of labels maps that match the bundles.')

    p.add_argument('--fitting_func', choices=['lin_up', 'lin_down', 'exp',
                                              'inv', 'log'],
                   help='Function to weigh points using their distance.'
                        '\n[Default: %(default)s]')

    p2 = p.add_argument_group(title='Visualization options')
    p3 = p2.add_mutually_exclusive_group()
    p3.add_argument('--show_rendering', action='store_true',
                    help='Display VTK window (optional).')
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
    p2.add_argument('--background', metavar=('R', 'G', 'B'), nargs=3,
                    default=[1, 1, 1], type=parser_color_type,
                    help='RBG values [0, 255] of the color of the background.'
                    '\n[Default: %(default)s]')

    add_reference_arg(p)
    add_json_args(p)
    add_overwrite_arg(p)

    return p


def fit_circle_2d(x, y, dist_w):
    """
    Least square for circle fitting in 2D
    dist_w allows for re-weighting of points
    """
    if dist_w is None:
        dist_w = np.ones(len(x))

    # Fit a circle in 2D using least-squares
    A = np.array([x, y, dist_w]).T
    b = x**2 + y**2
    params = np.linalg.lstsq(A, b, rcond=None)[0]

    # Get circle parameters from solution
    x_c = params[0]/2
    y_c = params[1]/2
    r = np.sqrt(params[2] + x_c**2 + y_c**2)

    return x_c, y_c, r


def rodrigues_rot(P, n0, n1):
    """
    Rodrigues rotation (not mine, see URL)
    - Rotate given points based on a starting and ending vector
    - Axis k and angle of rotation theta given by vectors n0,n1
    P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
    https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
    """

    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[np.newaxis, :]

    # Get vector of rotation k and angle theta
    n0 /= np.linalg.norm(n0)
    n1 /= np.linalg.norm(n1)
    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(n0, n1))

    # Compute rotated points
    P_rot = np.zeros((len(P), 3))
    for i in range(len(P)):
        P_rot[i] = P[i]*np.cos(theta) + np.cross(k, P[i]) * \
            np.sin(theta) + k*np.dot(k, P[i])*(1-np.cos(theta))

    return P_rot


def fit_circle_planar(pts, dist_w):
    # Fitting plane by SVD for the mean-centered data
    pts_mean = pts.mean(axis=0)
    pts_centered = pts - pts_mean

    _, _, V = svd(pts_centered, full_matrices=False)
    normal = V[2, :]

    # Project points to coords X-Y in 2D plane
    pts_xy = rodrigues_rot(pts_centered, normal, [0, 0, 1])

    # Fit circle in new 2D coords
    dist = np.linalg.norm(pts_centered, axis=1)
    if dist_w == 'lin_up':
        dist /= np.max(dist)
    elif dist_w == 'lin_down':
        dist /= np.max(dist)
        dist = 1 - dist
    elif dist_w == 'exp':
        dist /= np.max(dist)
        dist = np.exp(dist)
    elif dist_w == 'inv':
        dist /= np.max(dist)
        dist = 1 / dist
    elif dist_w == 'log':
        dist /= np.max(dist)
        dist = np.log(dist+1)
    else:
        dist = None

    x_c, y_c, radius = fit_circle_2d(pts_xy[:, 0], pts_xy[:, 1], dist)

    # Transform circle center back to 3D coords
    pts_recentered = rodrigues_rot(np.array([x_c, y_c, 0]),
                                   [0, 0, 1], normal) + pts_mean

    return pts_recentered, radius


def fit_circle_in_space(positions, directions, dist_w=None):
    # Project all points to a plane perpendicular to the centroid
    u_directions = np.average(directions, axis=0)
    u_directions /= np.linalg.norm(u_directions)
    barycenter = np.average(positions, axis=0)
    vector = positions - barycenter

    dist = np.zeros((len(vector)))
    proj_positions = np.zeros((len(vector), 3))
    for i in range(len(vector)):
        dist[i] = np.dot(vector[i], u_directions)
        proj_positions[i] = positions[i] - dist[i]*u_directions

    # With all points on a fixed plane, estimate a circle
    center, radius = fit_circle_planar(proj_positions, dist_w)
    dist = np.linalg.norm(proj_positions - center, axis=1)
    error = np.average(np.sqrt((dist - radius)**2))

    return center, radius, error


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # The number of labels maps must be equal to the number of bundles
    tmp = args.in_bundles + args.in_labels
    args.in_labels = args.in_bundles[(len(tmp) // 2):] + args.in_labels
    args.in_bundles = args.in_bundles[0:len(tmp) // 2]
    assert_inputs_exist(parser, args.in_bundles+args.in_labels)
    assert_output_dirs_exist_and_empty(parser, args, [],
                                       optional=args.save_rendering)

    stats = {}
    num_digits_labels = 3
    scene = window.Scene()
    scene.background(tuple(map(int, args.background)))
    for i, filename in enumerate(args.in_bundles):
        sft = load_tractogram_with_reference(parser, args, filename)
        sft.to_vox()
        sft.to_corner()
        img_labels = nib.load(args.in_labels[i])

        # same subject: same header or coregistered subjects: same header
        if not is_header_compatible(sft, args.in_bundles[0]) \
                or not is_header_compatible(img_labels, args.in_bundles[0]):
            parser.error('All headers must be identical.')

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
                                     order=0)
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
            c, d, e = fit_circle_in_space(labels_dict[key][0],
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
                = {'mean': float(radius[label-1])*2,
                   'std': float(error[label-1])}
        stats[bundle_name] = {'diameter': tmp_dict}

        if args.show_rendering or args.save_rendering:
            tube_actor = create_tube_with_radii(centroid_smooth, radius, error,
                                                wireframe=args.wireframe,
                                                error_coloring=args.error_coloring)
            scene.add(tube_actor)
            cmap = get_colormap('jet')
            coloring = cmap(pts_labels / np.max(pts_labels))[:, 0:3]
            streamlines_actor = actor.streamtube(sft.streamlines,
                                                 linewidth=args.width,
                                                 opacity=args.opacity,
                                                 colors=coloring)
            scene.add(streamlines_actor)

            slice_actor = actor.slicer(data_labels, np.eye(4))
            slice_actor.opacity(0.0)
            scene.add(slice_actor)

    # If there's actually streamlines to display
    if args.show_rendering:
        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()
    elif args.save_rendering:
        scene.reset_camera()
        snapshot(scene, os.path.join(args.save_rendering, 'superior.png'),
                 size=(1920, 1080), offscreen=True)

        scene.pitch(180)
        scene.reset_camera()
        snapshot(scene, os.path.join(args.save_rendering, 'inferior.png'),
                 size=(1920, 1080), offscreen=True)

        scene.pitch(90)
        scene.set_camera(view_up=(0, 0, 1))
        scene.reset_camera()
        snapshot(scene, os.path.join(args.save_rendering, 'posterior.png'),
                 size=(1920, 1080), offscreen=True)

        scene.pitch(180)
        scene.set_camera(view_up=(0, 0, 1))
        scene.reset_camera()
        snapshot(scene, os.path.join(args.save_rendering, 'anterior.png'),
                 size=(1920, 1080), offscreen=True)

        scene.yaw(90)
        scene.reset_camera()
        snapshot(scene, os.path.join(args.save_rendering, 'right.png'),
                 size=(1920, 1080), offscreen=True)

        scene.yaw(180)
        scene.reset_camera()
        snapshot(scene, os.path.join(args.save_rendering, 'left.png'),
                 size=(1920, 1080), offscreen=True)
    print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
