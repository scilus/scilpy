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
                                             dist_point_segment)
from scilpy.tracking.utils import tqdm_if_verbose
from scilpy.tractanalysis.todi import TrackOrientationDensityImaging


def mean_fibertube_density(sft):
    """
    Estimates the average per-voxel spatial density of a set of fibertubes.
    This is obtained by dividing the volume of fibertube segments present
    each voxel by the the total volume of the voxel.

    Parameters
    ----------
    sft: StatefulTractogram
        Stateful Tractogram object containing the fibertubes.

    Returns
    -------
    mean_density: float
        Per-voxel spatial density, averaged for the whole tractogram.
    """
    diameters = np.reshape(sft.data_per_streamline['diameters'],
                           len(sft.streamlines))
    mean_diameter = np.mean(diameters)

    mean_segment_lengths = []
    for streamline in sft.streamlines:
        mean_segment_lengths.append(
            np.mean(np.linalg.norm(streamline[1:] - streamline[:-1], axis=-1)))
    mean_segment_length = np.mean(mean_segment_lengths)
    # Computing mean tube density per voxel.
    sft.to_vox()
    # Because compute_todi expects streamline points (in voxel coordinates)
    # to be in the range [0, size] rather than [-0.5, size - 0.5], we shift
    # the voxel origin to corner.
    sft.to_corner()

    # Computing TDI
    _, data_shape, _, _ = sft.space_attributes
    todi_obj = TrackOrientationDensityImaging(tuple(data_shape))
    todi_obj.compute_todi(sft.streamlines)
    img = todi_obj.get_tdi()
    img = todi_obj.reshape_to_3d(img)

    nb_voxels_nonzero = np.count_nonzero(img)
    sum = np.sum(img, axis=-1)
    sum = np.sum(sum, axis=-1)
    sum = np.sum(sum, axis=-1)

    mean_seg_volume = np.pi * ((mean_diameter/2) ** 2) * mean_segment_length

    mean_seg_count = sum / nb_voxels_nonzero
    mean_volume = mean_seg_count * mean_seg_volume
    mean_density = mean_volume / (sft.voxel_sizes[0] *
                                  sft.voxel_sizes[1] *
                                  sft.voxel_sizes[2])

    return mean_density


def min_external_distance(sft, verbose):
    """"
    Calculates the minimal distance in between two fibertubes. A RuntimeError
    is thrown if a collision is detected (i.e. a negative distance is found).
    Use IntersectionFinder to remove intersections from fibertubes.

    Parameters
    ----------
    sft: StatefulTractogram
        Stateful Tractogram object containing the fibertubes
    verbose: bool
        Whether to make the function verbose.

    Returns
    -------
    min_external_distance: float
        Minimal distance found between two fibertubes.
    min_external_distance_vec: ndarray
        Vector representation of min_external_distance.
    """
    centerlines = sft.streamlines
    diameters = np.reshape(sft.data_per_streamline['diameters'],
                           len(centerlines))
    max_diameter = np.max(diameters)

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
                                          max_seg_length + max_diameter,
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
                raise RuntimeError(
                    'The input streamlines contained a collision after \n'
                    'filtering. This is unlikely to be an error of this \n'
                    'script, and instead may be due to your original data \n'
                    'using very high float precision. For more info on \n'
                    'this issue, please see the documentation for'
                    'scil_tractogram_filter_collisions.py.')

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
    seeds_fiber: ndarray
        Array containing the fiber index of each seed. If the seed is not in a
        fiber, its value will be -1.
    """
    seeds_fiber = [-1] * len(seeds)

    for si, seed in enumerate(seeds):
        for fi, fiber in enumerate(centerlines):
            if point_in_cylinder(fiber[0], fiber[1], diameters[fi]/2, seed):
                seeds_fiber[si] = fi
                break

    return np.array(seeds_fiber)


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

    # Building a tree for first fibertube
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
        neighbors = tree.query_parallel(streamline)[1]

        for pi, point in enumerate(streamline):
            nearest_index = neighbors[pi][0]

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
def endpoint_connectivity(blur_radius, centerlines, centerlines_length,
                          diameters, streamlines, seeds_fiber):
    """
    For every streamline, find whether or not it has reached the end segment
    of its fibertube. Each streamline is associated with an "Arrival fibertube
    segment", which is the closest fibertube segment to its before-last
    coordinate.

    IMPORTANT: Streamlines given as input to be scored should be forward-only,
    which means they are saved so that [0] is the seeding position and [-1] is
    the end.

    VC: "Valid Connection": A streamline whose arrival fibertube segment is
    the final segment of the fibertube in which is was originally seeded.

    IC: "Invalid Connection": A streamline whose arrival fibertube segment is
    the start or final segment of a fibertube in which is was not seeded.

    NC: "No Connection": A streamline whose arrival fibertube segment is
    not the start or final segment of any fibertube.

    Parameters
    ----------
    blur_radius: float
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
        streamline, contains the index of the fibertube in which it has been
        seeded.

    Return
    ------
    vc: list
        List containing the indices of all streamlines that are valid
        connections.
    ic: list
        List containing the indices of all streamlines that are invalid
        connections.
    nc: list
        List containing the indices of all streamlines that are no
        connections.
    """
    max_diameter = np.max(diameters)

    # objmode allows the execution of non numba-compatible code within a numba
    # function
    with objmode(centers='float64[:, :]', indices='int64[:, :]',
                 max_seg_length='float64'):
        centers, indices, max_seg_length = streamlines_to_segments(
            centerlines, False)

    tree = nbKDTree(centers)

    vc = set()
    ic = set()
    nc = set()

    # streamline[-2] is the last point with a valid direction
    all_neighbors = tree.query_radius(
        streamlines[:, -2], blur_radius + max_seg_length / 2 + max_diameter)

    for streamline_index, streamline in enumerate(streamlines):
        seed_fi = seeds_fiber[streamline_index]
        neighbors = all_neighbors[streamline_index]

        closest_dist = np.inf
        closest_seg = 0

        # Finding closest segment
        # There will always be a neighbor to override np.inf
        for segment_index in neighbors:
            fibertube_index = indices[segment_index][0]
            point_index = indices[segment_index][1]

            dist, _, _ = dist_point_segment(
                centerlines[fibertube_index][point_index],
                centerlines[fibertube_index][point_index+1],
                streamline[-2])

            if dist < closest_dist:
                closest_dist = dist
                closest_seg = segment_index

        fibertube_index = indices[closest_seg][0]
        point_index = indices[closest_seg][1]

        # If the closest segment is the last of its centerlines
        if point_index == centerlines_length[fibertube_index]-1:
            if fibertube_index == seed_fi:
                vc.add(streamline_index)
            else:
                ic.add(streamline_index)
        elif point_index == 0:
            ic.add(streamline_index)
        else:
            nc.add(streamline_index)

    return list(vc), list(ic), list(nc)
