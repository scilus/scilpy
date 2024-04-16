import numpy as np
from numba import objmode

from math import sqrt, acos
from numba import njit
from numba_kdtree import KDTree
from scipy.spatial.transform.rotation import Rotation
from scilpy.tracking.fibertube import (segment_tractogram,
                                       point_in_cylinder,
                                       dist_segment_segment,
                                       dist_point_segment,
                                       sphere_cylinder_intersection)
from scilpy.io.utils import v_enumerate


def min_external_distance(fibers, diameters, verbose):
    # Can be improved in speed by using Numba.
    # (See get_streamlines_as_fixed_array and numba-kdtree)
    seg_centers, seg_indices, max_seg_length = segment_tractogram(fibers,
                                                                  verbose)
    tree = KDTree(seg_centers)

    min_external_distance = np.float32('inf')
    min_external_distance_vec = np.zeros(0, dtype=np.float32)

    for segi, center in v_enumerate(seg_centers,
                                    verbose):
        si = seg_indices[segi][0]

        neighbors = tree.query_ball_point(center,
                                          max_seg_length,
                                          workers=-1)

        for neighbor_segi in neighbors:
            neighbor_si = seg_indices[neighbor_segi][0]

            # Skip if neighbor is our streamline
            if neighbor_si == si:
                continue

            p0 = fibers[si][seg_indices[segi][1]]
            p1 = fibers[si][seg_indices[segi][1] + 1]
            q0 = fibers[neighbor_si][seg_indices[neighbor_segi][1]]
            q1 = fibers[neighbor_si][seg_indices[neighbor_segi][1] + 1]

            rp = diameters[si] / 2
            rq = diameters[neighbor_si] / 2

            distance, vector, *_ = dist_segment_segment(p0, p1, q0, q1)
            external_distance = distance - rp - rq

            if external_distance < 0:
                raise RuntimeError('The input fibers contained a \n'
                                   'collision. Filter them prior \n'
                                   'to acquiring metrics.')

            if (external_distance < min_external_distance):
                min_external_distance = external_distance
                min_external_distance_vec = (
                    get_external_distance_vec(vector, rp, rq))

    return min_external_distance, min_external_distance_vec


def max_voxels(diagonal):
    max_voxel_anisotropic = np.abs(diagonal).astype(np.float32)

    # Find an isotropic voxel within the anisotropic one
    min_edge = min(max_voxel_anisotropic)
    max_voxel_isotropic = np.array([min_edge, min_edge, min_edge],
                                   dtype=np.float32)

    return (max_voxel_anisotropic, max_voxel_isotropic)


def true_max_voxel(diagonal):
    hyp = np.linalg.norm(diagonal)
    edge = hyp * sqrt(2)/2

    # The rotation should be such that the diagonal becomes aligned
    # with [1, 1, 1]
    diag = diagonal / np.linalg.norm(diagonal)
    dest = [1, 1, 1] / np.linalg.norm([1, 1, 1])

    v = np.cross(diag, dest)
    v /= np.linalg.norm(v)
    theta = acos(np.dot(diag, dest))
    rotation_matrix = Rotation.from_rotvec(v * theta)

    return (rotation_matrix, edge)


@njit
def get_external_distance_vec(vector, rp, rq):
    # Given a distance vector between two fiber centroids, find their
    # external distance
    unit_distance_vec = vector / np.linalg.norm(vector)
    external_distance_vec = (vector - rp * unit_distance_vec - rq *
                             unit_distance_vec)

    return external_distance_vec


@njit
def resolve_origin_seeding(seeds, fibers, diameters):
    """
    Associates given seeds to segment 0 of the fibertube in which they have
    been generated. This pairing only works with fiber origin seeding.

    Parameters
    ----------
    seeds: ndarray
    fibers: ndarray
        Fibertube centroids given as a fixed array
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
        for fi, fiber in enumerate(fibers):
            if point_in_cylinder(fiber[0], fiber[1], diameters[fi]/2, seed):
                seeds_fiber[si] = fi
                break

    return seeds_fiber


@njit
def mean_reconstruction_error(fibers, fibers_length, diameters, streamlines,
                              streamlines_length, seeds_fiber,
                              return_error_tractogram=False):
    """
    For each provided streamline, finds the mean distance between its
    coordinates and the fibertube it has been seeded in.

    Parameters
    ----------
    fibers: ndarray
        Fixed array containing ground-truth fibertube centroids.
    fibers_length: ndarray
        Fixed array containing the number of coordinates of each fibertube
        centroids.
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

    with objmode(centers='float64[:, :]', indices='int64[:, :]'):
        centers, indices, _ = segment_tractogram(fibers)
    centers_fixed_length = len(fibers[0])-1

    tree = KDTree(centers[:fibers_length[0]-1])
    tree_fi = 0

    for si, streamline_fixed in enumerate(streamlines):
        streamline = streamline_fixed[:streamlines_length[si]-1]
        errors = []

        seeded_fi = seeds_fiber[si]
        fiber = fibers[seeded_fi]
        radius = diameters[seeded_fi] / 2

        # Rebuild tree for current fiber.
        if tree_fi != seeded_fi:
            tree = KDTree(
                centers[centers_fixed_length * seeded_fi:
                        (centers_fixed_length * seeded_fi +
                         fibers_length[seeded_fi] - 1)])

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
                distance, vector, collFib, collStr = dist_point_segment(
                    pt1, pt2, point)
                errors.append(distance - radius)

                if return_error_tractogram:
                    vector /= np.linalg.norm(vector)
                    error_tractogram.append([collFib - vector * radius,
                                             collStr])

        mean_errors.append(np.array(errors).mean())

    return mean_errors, error_tractogram


@njit
def endpoint_connectivity(step_size, sampling_radius, fibers,
                          fibers_length, diameters, streamlines,
                          seeds_fiber, random_generator):
    """
    TODO: Particularly inefficient. To be improved with a KDTree.
    For every streamline, find whether or not it has reached the end segment
    of its fiber.

    VC: "Valid Connection": Contains streamlines that ended in the final
        segment of the fiber in which they have been seeded.
    IC: "Invalid Connection": Contains streamlines that ended in the final
        segment of another fiber.
    NC: "No Connection": Contains streamlines that have not ended in the final
        segment of any fiber.

    Parameters
    ----------
    step_size: any
    sampling_radius: any
    fibers: ndarray
        Fixed array containing ground-truth fibertube centroids.
    fibers_length: ndarray
        Fixed array containing the number of coordinates of each fibertube
        centroids.
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
    random_generator: any

    Return
    ------
    truth_vc: list
        Connections that are valid at ground-truth resolution.
    truth_ic: list
    truth_nc: list
    resolution_vc: list
        Connections that are valid at degraded resolution.
    resolution_ic: list
    resolution_nc: list
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
        for fi, fiber in enumerate(fibers):
            fib_pt1 = fiber[fibers_length[fi] - 2]
            fib_pt2 = fiber[fibers_length[fi] - 1]
            radius = diameters[fi] / 2

            # Check all the points of the estimated last segment of streamline
            estimated_overstep_mm = (radius + sampling_radius + step_size)
            estimated_streamline_last_seg_nb_pts = int(
                (np.linalg.norm(fib_pt2-fib_pt1) +
                 estimated_overstep_mm) // step_size)

            # Streamlines are reversed. streamline[:n] gives the last n points
            for point in streamline[:estimated_streamline_last_seg_nb_pts]:
                seed_fi = seeds_fiber[si]

                if point_in_cylinder(fib_pt1, fib_pt2, radius, point):
                    truth_connected = True
                    if fi == seed_fi:
                        truth_vc.append(si)
                    else:
                        truth_ic.append((si, fi))

                volume, _ = sphere_cylinder_intersection(
                        point, sampling_radius,
                        fib_pt1, fib_pt2, radius,
                        1000, random_generator)

                if volume != 0:
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
