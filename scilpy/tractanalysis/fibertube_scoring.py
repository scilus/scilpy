import numpy as np
import nibabel as nib
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


def fibertube_density(sft, samples_per_voxel_axis, verbose=False):
    """
    Estimates the per-voxel volumetric density of a set of fibertubes. In other
    words, how much space is occupied by fibertubes and how much is emptiness.

    Works by building a binary mask segmenting voxels that contain at least
    a single fibertube. Then, valid voxels are finely sampled and we count the
    number of samples that landed within a fibertube. For each voxel, this
    number is then divided by its total amount of samples.

    Parameters
    ----------
    sft: StatefulTractogram
        Stateful Tractogram object containing the fibertubes.
    samples_per_voxel_axis: int
        Number of samples to be created along a single axis of a voxel. The
        total number of samples in the voxel will be this number cubed.
    verbose: bool
        Whether the function and sub-functions should be verbose.

    Returns
    -------
    density_grid: ndarray
        Per-voxel volumetric density of fibertubes as a 3D image.
    density_flat: list
        Per-voxel volumetric density of fibertubes as a list, containing only
        the voxels that were valid in the binary mask (i.e. that contained
        fibertubes). This is useful for calculating measurements on the
        various density values, like mean, median, etc.
    collision_grid: ndarray
        Per-voxel density of fibertube collisions.
    collision_flat: list
        Per-voxel density of fibertubes collisions as a list, containing only
        the voxels that were valid in the binary mask (i.e. that contained
        fibertubes). This is useful for calculating measurements on the
        various density values, like mean, median, etc.
    """
    if "diameters" not in sft.data_per_streamline:
        raise ValueError('No diameter found as data_per_streamline '
                         'in the provided tractogram')
    diameters = np.reshape(sft.data_per_streamline['diameters'],
                           len(sft.streamlines))
    max_diameter = np.max(diameters)

    # Everything will be in vox for simplicity.
    sft.to_vox()
    # Building a binary mask using TODI
    # Because compute_todi expects streamline points (in voxel coordinates)
    # to be in the range [0, size] rather than [-0.5, size - 0.5], we shift
    # the voxel origin to corner.
    sft.to_corner()
    _, data_shape, _, _ = sft.space_attributes
    todi_obj = TrackOrientationDensityImaging(tuple(data_shape))
    todi_obj.compute_todi(sft.streamlines, False)
    mask = todi_obj.get_mask()
    mask = todi_obj.reshape_to_3d(mask)

    sampling_density = np.array([samples_per_voxel_axis,
                                 samples_per_voxel_axis,
                                 samples_per_voxel_axis])

    # Source: dipy.tracking.seeds_from_mask
    # Grid of points between -.5 and .5, centered at 0, with given density
    grid = np.mgrid[0: sampling_density[0], 0: sampling_density[1],
                    0: sampling_density[2]]
    grid = grid.T.reshape((-1, 3))
    grid = grid / sampling_density
    grid += 0.5 / sampling_density - 0.5
    grid = grid.reshape(*sampling_density, 3)

    # Back to corner origin
    grid += 0.5

    # Add samples to each voxel in mask
    samples = np.empty(mask.shape, dtype=object)
    for i, j, k in np.ndindex(mask.shape):
        if mask[i][j][k]:
            samples[i][j][k] = [i, j, k] + grid

    # Building KDTree from fibertube segments
    centers, indices, max_seg_length = streamlines_to_segments(
        sft.streamlines, verbose)
    tree = KDTree(centers)

    density_grid = np.zeros(mask.shape)
    density_flat = []
    collision_grid = np.zeros(mask.shape)
    collision_flat = []
    # For each voxel, get density
    for i, j, k in tqdm_if_verbose(np.ndindex(mask.shape), verbose,
                                   total=len(np.ravel(mask))):
        if not mask[i][j][k]:
            continue

        voxel_samples = np.reshape(samples[i][j][k], (-1, 3))

        # Returns an list of lists of neighbor indexes for each sample
        # Ex: [[265, 45, 0, 1231], [12, 67]]
        all_sample_neighbors = tree.query_ball_point(
            voxel_samples, max_seg_length/2+max_diameter/2, workers=-1)

        nb_samples_in_fibertubes = 0
        # Set containing sets of fibertube indexes
        # This way, each pair of fibertube is only entered once.
        # This is unless there is a trio. Then the same colliding fibertubes
        # may be entered more than once.
        collisions = set()
        for index, sample in enumerate(voxel_samples):
            neigbors = all_sample_neighbors[index]
            fibertubes_touching_sample = set()
            for segi in neigbors:
                fi = indices[segi][0]
                pi = indices[segi][1]
                centerline = sft.streamlines[fi]
                radius = diameters[fi] / 2

                dist, _, _ = dist_point_segment(centerline[pi],
                                                centerline[pi+1],
                                                np.float32(sample))

                if dist < radius:
                    fibertubes_touching_sample.add(fi)

            if len(fibertubes_touching_sample) > 0:
                nb_samples_in_fibertubes += 1
            if len(fibertubes_touching_sample) > 1:
                collisions.add(frozenset(fibertubes_touching_sample))

        density_grid[i][j][k] = nb_samples_in_fibertubes / len(voxel_samples)
        density_flat.append(density_grid[i][j][k])
        collision_grid[i][j][k] = len(collisions) / len(voxel_samples)
        collision_flat.append(collision_grid[i][j][k])

    return density_grid, density_flat, collision_grid, collision_flat


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
        streamline, contains the index of the fiber in which it has been
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

    for si, streamline in enumerate(streamlines):
        seed_fi = seeds_fiber[si]

        # streamline[1] is the last point with a valid direction
        neighbors = tree.query_radius(
            streamline[1], blur_radius + max_seg_length / 2 + max_diameter)[0]

        min_dist = np.inf
        min_seg = 0

        # Finding closest segment
        # There will always be a neighbor to override np.inf
        for neighbor_segi in neighbors:
            fi = indices[neighbor_segi][0]
            pi = indices[neighbor_segi][1]

            dist, _, _ = dist_point_segment(centerlines[fi][pi],
                                            centerlines[fi][pi+1],
                                            streamline[1])

            if dist < min_dist:
                min_dist = dist
                min_seg = neighbor_segi

        fi = indices[min_seg][0]
        pi = indices[min_seg][1]

        # If the closest segment is the last of its centerlines
        if pi == centerlines_length[fi]-1:
            if fi == seed_fi:
                vc.add(si)
            else:
                ic.add(si)
        elif pi == 0:
            ic.add(si)
        else:
            nc.add(si)

    return list(vc), list(ic), list(nc)
