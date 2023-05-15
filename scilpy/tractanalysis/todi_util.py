# -*- coding: utf-8 -*-
import logging

import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree
from scipy.sparse import bsr_matrix


def _subdivide_streamline(streamline, n_steps):
    if n_steps < 2:
        return streamline

    dirs = streamline[1:] - streamline[:-1]
    subdivided = np.zeros((n_steps * (len(streamline) - 1) + 1, 3))
    subdivided[::n_steps] = streamline
    for s in range(1, n_steps):
        subdivided[s::n_steps] = streamline[:-1] + s / n_steps * dirs

    return subdivided


def streamlines_to_segments(streamlines, n_steps=1):
    """Split streamlines into its segments.

    Parameters
    ----------
    streamlines : list of numpy.ndarray
        List of streamlines.

    Returns
    -------
    segments : numpy.ndarray (2D)
        Segments array representation with the first and last points.
    """
    vts_0_list = []
    vts_1_list = []
    for streamline in streamlines:
        streamline = _subdivide_streamline(streamline, n_steps)
        vts_0_list.append(streamline[:-1])
        vts_1_list.append(streamline[1:])

    segments = np.stack((np.vstack(vts_0_list), np.vstack(vts_1_list)), axis=0)
    return segments


def streamlines_to_endpoints(streamlines):
    """Equivalent to streamlines resampling to 2 points (first and last).

    Parameters
    ----------
    streamlines : list of numpy.ndarray
        List of streamlines.

    Returns
    -------
    endpoints : numpy.ndarray (2D)
        Endpoint array representation with the first and last points.
    """
    endpoints = np.zeros((2, len(streamlines), 3))
    for i, streamline in enumerate(streamlines):
        endpoints[0, i] = streamline[0]
        endpoints[1, i] = streamline[-1]

    return endpoints


def streamlines_to_pts_dir_norm(streamlines, n_steps=1, asymmetric=False):
    """Evaluate each segment: mid position, direction, length.

    Parameters
    ----------
    streamlines :  list of numpy.ndarray
        List of streamlines.

    Returns
    -------
    seg_mid : numpy.ndarray (2D)
        Mid position (x,y,z) of all streamlines' segments.
    seg_dir : numpy.ndarray (2D)
        Direction (x,y,z) of all streamlines' segments.
    seg_norm : numpy.ndarray (2D)
        Length of all streamlines' segments.
    """
    segments = streamlines_to_segments(streamlines, n_steps)
    seg_mid = get_segments_mid_pts_positions(segments)
    seg_dir, seg_norm = get_segments_dir_and_norm(segments,
                                                  seg_mid,
                                                  asymmetric)

    mask = seg_norm > 1.0e-20
    if ~mask.any():
        logging.warning("WARNING : There is at least one streamline with "
                        "overlapping points in the tractogram.")

    return seg_mid[mask], seg_dir[mask], seg_norm[mask]


def get_segments_mid_pts_positions(segments):
    return 0.5 * (segments[0] + segments[1])


def get_segments_vectors(segments):
    return segments[1] - segments[0]


def get_segments_dir_and_norm(segments, seg_mid=None, asymmetric=False):
    if asymmetric:
        seg_vecs = get_segments_vectors(segments)
        return get_vectors_dir_and_norm_rel_to_center(seg_vecs, seg_mid)
    return get_vectors_dir_and_norm(get_segments_vectors(segments))


def get_vectors_dir_and_norm(vectors):
    vectors_norm = compute_vectors_norm(vectors)
    vectors_dir = vectors / vectors_norm.reshape((-1, 1))
    return vectors_dir, vectors_norm


def get_vectors_dir_and_norm_rel_to_center(vectors, seg_mid_pts):
    """ Evaluates vectors direction and norm by taking into account the
        orientation and position of segments in relation to the center
        of voxel
    """
    vectors_norm = compute_vectors_norm(vectors)
    vectors_dir = vectors / vectors_norm.reshape((-1, 1))

    # we create an array of voxel centers for each of our points
    vox_centers = seg_mid_pts.astype(int) + 0.5

    # directions to center of voxel for each segment
    dir_to_center = (vox_centers - seg_mid_pts).flatten()
    r, c = (vectors_dir.shape[0], 3 * vectors_dir.shape[0])
    rows = np.arange(r).repeat(3)
    cols = np.arange(c)
    dir_to_center_mat = bsr_matrix((dir_to_center, (rows, cols)), shape=(r, c))

    # compute dot product between direction of vectors and direction to center
    dots = dir_to_center_mat.dot(vectors_dir.flatten()).reshape((-1, 1))

    # when dot is greater that 0, the vector goes toward the center
    # of the voxel we flip the direction of such vectors
    vectors_dir_rel = np.where(dots > 0, -vectors_dir, vectors_dir)

    return vectors_dir_rel, vectors_norm


def psf_from_sphere(sphere_vertices):
    return np.abs(np.dot(sphere_vertices, sphere_vertices.T))


# Mask functions
def generate_mask_indices_1d(nb_voxel, indices_1d):
    mask_1d = np.zeros(nb_voxel, dtype=bool)
    mask_1d[indices_1d] = True
    return mask_1d


def get_indices_1d(volume_shape, pts):
    return np.ravel_multi_index(pts.T.astype(int), volume_shape)


def get_dir_to_sphere_id(vectors, sphere_vertices):
    """Find the closest vector on the sphere vertices using a cKDT tree
        sphere_vertices must be normed (or all with equal norm).

    Parameters
    ----------
    vectors : numpy.ndarray (2D)
        Vectors representing the direction (x,y,z) of segments.
    sphere_vertices : numpy.ndarray (2D)
        Vertices of a Dipy sphere object.

    Returns
    -------
    dir_sphere_id : numpy.ndarray (1D)
        Sphere indices of the closest sphere direction for each vector
    """
    sphere_kdtree = cKDTree(sphere_vertices)
    _, dir_sphere_id = sphere_kdtree.query(vectors, k=1, workers=-1)
    return dir_sphere_id


# Generic Functions (vector norm)
def compute_vectors_norm(vectors):
    return norm(vectors, ord=2, axis=-1)


def normalize_vectors(vectors):
    return p_normalize_vectors(vectors, 2)


def p_normalize_vectors(vectors, p):
    return vectors / norm(vectors, ord=p, axis=-1, keepdims=True)
