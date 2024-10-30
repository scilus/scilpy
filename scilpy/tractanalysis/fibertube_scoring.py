import numpy as np
from numba import objmode

from math import sqrt, acos
from numba import njit
from numba_kdtree import KDTree as nbKDTree
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scilpy.tracking.fibertube_utils import (streamlines_to_segments,
                                             point_in_cylinder,
                                             dist_segment_segment,
                                             dist_point_segment,
                                             sphere_cylinder_intersection)
from scilpy.tracking.utils import tqdm_if_verbose


def min_external_distance(centerlines, diameters, verbose):
    """"
    Calculates the minimal distance in between two fibertubes. A RuntimeError
    is thrown if a collision is detected (i.e. a negative distance is found).
    Use IntersectionFinder to remove intersections from fibertubes.

    Parameters
    ----------
    centerlines: ndarray
        Centerlines of the fibertubes.
    diameters: ndarray
        Diameters of the fibertubes.
    verbose: bool
        Whether to make the function verbose.

    Returns
    -------
    min_external_distance: float
        Minimal distance found between two fibertubes.
    min_external_distance_vec: ndarray
        Vector representation of min_external_distance.
    """
    if len(centerlines) <= 1:
        ValueError("Cannot compute metrics of a tractogram with a single" +
                   "streamline or less")
    seg_centers, seg_indices, max_seg_length = streamlines_to_segments(
        centerlines, verbose)
    tree = KDTree(seg_centers)
    min_external_distance = np.inf
    min_external_distance_vec = np.zeros(0, dtype=np.float32)

    for segi, center in tqdm_if_verbose(enumerate(seg_centers), verbose,
                                        total=len(seg_centers)):
        si = seg_indices[segi][0]

        neighbors = tree.query_ball_point(center,
                                          max_seg_length,
                                          workers=-1)

        for neighbor_segi in neighbors:
            neighbor_si = seg_indices[neighbor_segi][0]

            # Skip if neighbor is our streamline
            if neighbor_si == si:
                continue

            p0 = centerlines[si][seg_indices[segi][1]]
            p1 = centerlines[si][seg_indices[segi][1] + 1]
            q0 = centerlines[neighbor_si][seg_indices[neighbor_segi][1]]
            q1 = centerlines[neighbor_si][seg_indices[neighbor_segi][1] + 1]

            rp = diameters[si] / 2
            rq = diameters[neighbor_si] / 2

            distance, vector, *_ = dist_segment_segment(p0, p1, q0, q1)
            external_distance = distance - rp - rq

            if external_distance < 0:
                raise RuntimeError('The input fibertubes contained a \n'
                                   'collision. Filter them prior \n'
                                   'to acquiring metrics.')

            if (external_distance < min_external_distance):
                min_external_distance = external_distance
                min_external_distance_vec = (
                    get_external_vector_from_centerline_vector(vector, rp, rq)
                    )

    return min_external_distance, min_external_distance_vec


def max_voxels(diagonal):
    """
    Given the vector representing the smallest distance between two
    fibertubes, calculates the maximum sized voxels (anisotropic & isotropic)
    without causing any partial-volume effect.

    These voxel are expressed in the current 3D referential and are
    often rendered meaningless by it. See function max_voxel_rotated for an
    alternative.

    Parameters
    ----------
    diagonal: ndarray
        Vector representing the smallest distance between two
        fibertubes.

    Returns
    -------
    max_voxel_anisotropic: ndarray
        Maximum sized anisotropic voxel.
    max_voxel_isotropic: ndarray
        Maximum sized isotropic voxel.
    """
    max_voxel_anisotropic = np.abs(diagonal).astype(np.float32)

    # Find an isotropic voxel within the anisotropic one
    min_edge = min(max_voxel_anisotropic)
    max_voxel_isotropic = np.array([min_edge, min_edge, min_edge],
                                   dtype=np.float32)

    return (max_voxel_anisotropic, max_voxel_isotropic)


def max_voxel_rotated(diagonal):
    """
    Given the vector representing the smallest distance between two
    fibertubes, calculates the maximum sized voxel without causing any
    partial-volume effect. This voxel is isotropic.

    This voxel is not expressed in the current 3D referential. It will require
    the tractogram to be rotated according to rotation_matrix for this voxel
    to be applicable.

    Parameters
    ----------
    diagonal: ndarray
        Vector representing the smallest distance between two
        fibertubes.

    Returns
    -------
    rotation_matrix: ndarray
        3x3 rotation matrix to be applied to the tractogram to align it with
        the voxel
    edge: float
        Edge size of the max_voxel_rotated.
    """
    hyp = np.linalg.norm(diagonal)
    edge = hyp / 3*sqrt(3)

    # The rotation should be such that the diagonal becomes aligned
    # with [1, 1, 1]
    diag = diagonal / np.linalg.norm(diagonal)
    dest = [1, 1, 1] / np.linalg.norm([1, 1, 1])

    v = np.cross(diag, dest)
    v /= np.linalg.norm(v)
    theta = acos(np.dot(diag, dest))
    rotation_matrix = Rotation.from_rotvec(v * theta).as_matrix()

    return (rotation_matrix, edge)


@njit
def get_external_vector_from_centerline_vector(vector, r1, r2):
    """
    Given a vector separating two fibertube centerlines, finds a
    vector that separates them from outside their diameter.

    Parameters
    ----------
    vector: ndarray
        Vector between two fibertube centerlines.
    rp: ndarray
        Radius of one of the fibertubes.
    rq: ndarray
        Radius of the other fibertube.

    Results
    -------
    external_vector: ndarray
        Vector between the two fibertubes, outside their diameter.
    """
    unit_vector = vector / np.linalg.norm(vector)
    external_vector = (vector - r1 * unit_vector - r2 *
                       unit_vector)

    return external_vector


@njit
def resolve_origin_seeding(seeds, centerlines, diameters):
    """
    Associates given seeds to segment 0 of the fibertube in which they have
    been generated. This pairing only works with fiber origin seeding.

    Parameters
    ----------
    seeds: ndarray
    centerlines: ndarray
        Fibertube centerlines given as a fixed array
        (see streamlines_as_fixed_array).
    diameters: ndarray

    Return
    ------
    seeds_fiber: list
        List containing the fiber index of each seed. If the seed is not in a
        fiber, its value will be -1.
    """
    seeds_fiber = [-1] * len(seeds)

    for si, seed in enumerate(seeds):
        for fi, fiber in enumerate(centerlines):
            if point_in_cylinder(fiber[0], fiber[1], diameters[fi]/2, seed):
                seeds_fiber[si] = fi
                break

    return seeds_fiber


@njit
def mean_reconstruction_error(centerlines, centerlines_length, diameters,
                              streamlines, streamlines_length, seeds_fiber,
                              return_error_tractogram=False):
    """
    For each provided streamline, finds the mean distance between its
    coordinates and the fibertube it has been seeded in.

    Parameters
    ----------
    centerlines: ndarray
        Fixed array containing ground-truth fibertube centerlines.
    centerlines_length: ndarray
        Fixed array containing the number of coordinates of each fibertube
        centerlines.
    diameters: list,
        Diameters of the fibertubes
    streamlines: ndarray
        Fixed array containing streamlines resulting from the tracking
        process.
    streamlines_length: ndarray,
        Fixed array containing the number of coordinates of each streamline
    seeds_fiber: list
        Array of the same length as there are streamlines. For every
        streamline, contains the index of the fiber in which it has been
        seeded.
    return_error_tractogram: bool = False

    Return
    ------
    mean_errors: list
        Array containing the mean error for every streamline.
    error_tractogram: list
        Empty when return_error_tractogram is set to False. Otherwise,
        contains a visual representation of the error between every streamline
        and the fiber in which it has been seeded.
    """
    mean_errors = []
    error_tractogram = []

    # objmode allows the execution of non numba-compatible code within a numba
    # function
    with objmode(centers='float64[:, :]', indices='int64[:, :]'):
        centers, indices, _ = streamlines_to_segments(centerlines, False)
    centers_fixed_length = len(centerlines[0])-1

    tree = nbKDTree(centers[:centerlines_length[0]-1])
    tree_fi = 0

    for si, streamline_fixed in enumerate(streamlines):
        streamline = streamline_fixed[:streamlines_length[si]-1]
        errors = []

        seeded_fi = seeds_fiber[si]
        fiber = centerlines[seeded_fi]
        radius = diameters[seeded_fi] / 2

        # Rebuild tree for current fiber.
        if tree_fi != seeded_fi:
            tree = nbKDTree(
                centers[centers_fixed_length * seeded_fi:
                        (centers_fixed_length * seeded_fi +
                         centerlines_length[seeded_fi] - 1)])

        # Querying nearest neighbor for each coordinate of the streamline.
        neighbor_indices = tree.query_parallel(streamline)[1]

        for pi, point in enumerate(streamline):
            nearest_index = neighbor_indices[pi][0]

            # Retrieving the closest cylinder segment.
            _, pi = indices[nearest_index]
            pt1 = fiber[pi]
            pt2 = fiber[pi + 1]

            # If we're within the fiber, error = 0
            if (np.linalg.norm(point - pt1) < radius or
                np.linalg.norm(point - pt2) < radius or
                    point_in_cylinder(pt1, pt2, radius, point)):
                errors.append(0.)
            else:
                distance, vector, segment_collision_point = dist_point_segment(
                    pt1, pt2, point)
                errors.append(distance - radius)

                if return_error_tractogram:
                    fiber_collision_point = segment_collision_point - (
                        vector / np.linalg.norm(vector)) * radius
                    error_tractogram.append([fiber_collision_point, point])

        mean_errors.append(np.array(errors).mean())

    return mean_errors, error_tractogram


@njit
def endpoint_connectivity(step_size, blur_radius, centerlines,
                          centerlines_length, diameters, streamlines,
                          seeds_fiber, random_generator):
    """
    For every streamline, find whether or not it has reached the end segment
    of its fiber.

    VC: "Valid Connection": Contains streamlines that ended in the final
        segment of the fiber in which they have been seeded.
    IC: "Invalid Connection": Contains streamlines that ended in the final
        segment of another fiber.
    NC: "No Connection": Contains streamlines that have not ended in the final
        segment of any fiber.

    TODO: Particularly inefficient. Could be improved with a KDTree.

    Parameters
    ----------
    step_size: any
        Step_size used during fibertube tracking.
    blur_radius: any
        Blur radius used during fibertube tracking.
    centerlines: ndarray
        Fixed array containing ground-truth fibertube centerlines.
    centerlines_length: ndarray
        Fixed array containing the number of coordinates of each fibertube
        centerlines.
    diameters: list,
        Diameters of the fibertubes.
    streamlines: ndarray
        Fixed array containing streamlines resulting from the tracking
        process.
    streamlines_length: ndarray,
        Fixed array containing the number of coordinates of each streamline
    seeds_fiber: list
        Array of the same length as there are streamlines. For every
        streamline, contains the index of the fiber in which it has been
        seeded.
    random_generator: any
        Random generator used to generate samples for estimating the volume of
        intersection between blurring sphere and fibertube segments.

    Return
    ------
    truth_vc: list
        Connections that are valid at ground-truth resolution.
    truth_ic: list
        Connections that are invalid at ground-truth resolution.
    truth_nc: list
        No-connections at ground-truth resolution.
    resolution_vc: list
        Connections that are valid at simulated resolution.
    resolution_ic: list
        Connections that are invalid at simulated resolution.
    resolution_nc: list
        No-connections at simulated resolution.
    """
    truth_vc = []
    truth_ic = []
    truth_nc = []
    res_vc = []
    res_ic = []
    res_nc = []
    for si, streamline in enumerate(streamlines):
        truth_connected = False
        res_connected = False
        for fi, fiber in enumerate(centerlines):
            fib_end_pt1 = fiber[centerlines_length[fi] - 2]
            fib_end_pt2 = fiber[centerlines_length[fi] - 1]
            radius = diameters[fi] / 2

            # Check all the points of the estimated last segment of streamline
            estimated_overstep_mm = (radius + blur_radius + step_size)
            estimated_streamline_last_seg_nb_pts = int(
                (np.linalg.norm(fib_end_pt2-fib_end_pt1) +
                 estimated_overstep_mm) // step_size)

            # Streamlines are reversed. streamline[:n] gives the last n points
            for point in streamline[:estimated_streamline_last_seg_nb_pts]:
                seed_fi = seeds_fiber[si]

                # Is in start segment of a fibertube and not ours
                if point_in_cylinder(fiber[0], fiber[1],
                                     radius, point) and fi != seed_fi:
                    truth_connected = True
                    truth_ic.append((si, fi))

                # Is in end segment of a fibertube
                if point_in_cylinder(fib_end_pt1, fib_end_pt2, radius, point):
                    truth_connected = True
                    if fi == seed_fi:
                        truth_vc.append(si)
                    else:
                        truth_ic.append((si, fi))

                # Passes by start segment of a fibertube and not ours
                volume_start, _ = sphere_cylinder_intersection(
                        point, blur_radius,
                        fiber[0], fiber[1], radius,
                        1000, random_generator)
                if volume_start != 0 and fi != seed_fi:
                    res_connected = True
                    res_ic.append((si, fi))

                # Passes by end segment of a fibertube
                volume_end, _ = sphere_cylinder_intersection(
                        point, blur_radius,
                        fib_end_pt1, fib_end_pt2, radius,
                        1000, random_generator)
                if volume_end != 0:
                    res_connected = True
                    if fi == seed_fi:
                        res_vc.append(si)
                    else:
                        res_ic.append((si, fi))

                if truth_connected or res_connected:
                    break

        if not truth_connected:
            truth_nc.append(si)
        if not res_connected:
            res_nc.append(si)

    return truth_vc, truth_ic, truth_nc, res_vc, res_ic, res_nc
