#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize bundles.

Example usages:

# Visualize streamlines as tubes, each bundle with a different color
$ scil_visualize_bundles.py path_to_bundles/ --shape tube \
    --random_coloring 1337

# Visualize a tractogram with each streamlines drawn as lines, colored with
# their local orientation, but only load 1 in 10 streamlines
$ scil_visualize_bundles.py tractogram.trk --shape line --subsample 10

# Visualize CSTs as large tubes and color them from a list of colors in a file
$ scil_visualize_bundles.py path_to_bundles/CST_* --width 0.5 \
    --color_dict colors.json
"""

import argparse
import json
import os

from dipy.io.utils import is_header_compatible
from fury import window, actor
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.linalg import svd
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import vtk

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_json_args,
                             assert_inputs_exist,
                             parser_color_type)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundles', nargs='+',
                   help='List of tractography files supported by nibabel.')
    p.add_argument('in_labels', nargs='+',
                   help='List of tractography files supported by nibabel.')

    p2 = p.add_argument_group(title='Visualization options')
    p2.add_argument('--show_rendering', action='store_true',
                    help='Display VTK window.')
    p2.add_argument('--wireframe', action='store_true',
                    help='Use wireframe for the tube.')
    p2.add_argument('--error_coloring', action='store_true',
                    help='Use the fitting error to color the tube.')
    p2.add_argument('--width', type=float, default=0.2,
                    help='Width of tubes or lines representing streamlines'
                    '\n[Default: %(default)s]')
    p2.add_argument('--opacity', type=int, default=0.1,
                    help='Opacity for the streamlines rendered with the tube.'
                    '\n[Default: %(default)s]')
    p2.add_argument('--background', metavar=('R', 'G', 'B'), nargs=3,
                    default=[0, 0, 0], type=parser_color_type,
                    help='RBG values [0, 255] of the color of the background.'
                    '\n[Default: %(default)s]')

    add_reference_arg(p)
    add_json_args(p)
    add_overwrite_arg(p)

    return p


def create_tube(positions, radii, error, error_coloring=False,
                wireframe=False):
    # Generate the polydata from the centroids
    joint_count = len(positions)
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(joint_count)
    for j in range(joint_count):
        pts.InsertPoint(j, positions[j])
        lines.InsertCellPoint(j)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(pts)
    polydata.SetLines(lines)

    # Generate the coloring from either the labels or the fitting error
    colors_arr = vtk.vtkFloatArray()
    for i in range(joint_count):
        if error_coloring:
            colors_arr.InsertNextValue(error[i])
        else:
            colors_arr.InsertNextValue(len(error) - 1 - i)
    colors_arr.SetName("colors")
    polydata.GetPointData().AddArray(colors_arr)

    # Generate the radii array for VTK
    radii_arr = vtk.vtkFloatArray()
    for i in range(joint_count):
        radii_arr.InsertNextValue(radii[i])
    radii_arr.SetName("radii")
    polydata.GetPointData().SetScalars(radii_arr)

    # Tube filter for the rendering with varying radii
    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetInputData(polydata)
    tubeFilter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tubeFilter.SetNumberOfSides(25)
    tubeFilter.CappingOn()

    # Map the coloring to the tube filter
    tubeFilter.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tubeFilter.GetOutputPort())
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("colors")
    if error_coloring:
        mapper.SetScalarRange(0, max(error))
    else:
        mapper.SetScalarRange(0, len(error))

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if wireframe:
        actor.GetProperty().SetRepresentationToWireframe()

    return actor


def fit_circle_2d(x, y):
    # Fit a circle in 2D using least-squares
    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2

    # Solve by method of least squares
    params = np.linalg.lstsq(A, b, rcond=None)[0]

    # Get circle parameters from solution
    x_c = params[0]/2
    y_c = params[1]/2
    r = np.sqrt(params[2] + x_c**2 + y_c**2)

    return x_c, y_c, r


def rodrigues_rot(P, n0, n1):
    """
    Rodrigues rotation (not mine)
    - Rotate given points based on a starting and ending vector
    - Axis k and angle of rotation theta given by vectors n0,n1
    P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
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


def fit_circle_planar(pts):
    # Fitting plane by SVD for the mean-centered data
    pts_mean = pts.mean(axis=0)
    pts_centered = pts - pts_mean

    _, _, V = svd(pts_centered, full_matrices=False)
    normal = V[2, :]

    # Project points to coords X-Y in 2D plane
    pts_xy = rodrigues_rot(pts_centered, normal, [0, 0, 1])

    # Fit circle in new 2D coords
    x_c, y_c, radius = fit_circle_2d(pts_xy[:, 0], pts_xy[:, 1])

    # Transform circle center back to 3D coords
    pts_recentered = rodrigues_rot(np.array([x_c, y_c, 0]),
                                   [0, 0, 1], normal) + pts_mean

    return pts_recentered, radius


def fit_circle_in_space(positions, directions):
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
    center, radius = fit_circle_planar(proj_positions)
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

        pts_labels = map_coordinates(data_labels, sft.streamlines._data.T-0.5,
                                     order=0)
        counter = 0
        labels_dict = {label: ([], []) for label in unique_labels}
        for streamline in sft.streamlines:
            direction = np.gradient(streamline, axis=0).tolist()
            curr_labels = pts_labels[counter:counter+len(streamline)].tolist()

            for i, label in enumerate(curr_labels):
                if label > 0:
                    labels_dict[label][0].append(streamline[i])
                    labels_dict[label][1].append(direction[i])

            counter += len(streamline)

        centroid = np.zeros((len(unique_labels), 3))
        diameter = np.zeros((len(unique_labels), 1))
        error = np.zeros((len(unique_labels), 1))
        for key in unique_labels:
            key = int(key)
            c, d, e = fit_circle_in_space(labels_dict[key][0],
                                          labels_dict[key][1])
            centroid[key-1], diameter[key-1], error[key-1] = c, d, e

        # Spatial smoothing to avoid degenerate estimation
        centroid_smooth = gaussian_filter(centroid, sigma=[1, 0],
                                          mode='nearest')
        centroid_smooth[::len(centroid)-1] = centroid[::len(centroid)-1]
        diameter = gaussian_filter(diameter, sigma=1, mode='nearest')
        error = gaussian_filter(error, sigma=1, mode='nearest')

        tmp_dict = {}
        for label in unique_labels:
            tmp_dict['{}'.format(label).zfill(num_digits_labels)] \
                = {'mean': float(diameter[label-1]),
                   'std': float(error[label-1])}
        stats[bundle_name] = {'diameter': tmp_dict}

        if args.show_rendering:
            tube_actor = create_tube(centroid_smooth, diameter, error,
                                     wireframe=args.wireframe,
                                     error_coloring=args.error_coloring)
            scene.add(tube_actor)
            cmap = plt.get_cmap('jet')
            coloring = cmap(pts_labels / np.max(pts_labels))[:, 0:3]
            streamlines_actor = actor.streamtube(sft.streamlines,
                                                 linewidth=args.width,
                                                 opacity=args.opacity,
                                                 colors=coloring)
            scene.add(streamlines_actor)

    # If there's actually streamlines to display
    if args.show_rendering:
        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()
    print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
