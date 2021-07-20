# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm
from scipy.spatial.ckdtree import cKDTree


def streamlines_to_segments(streamlines):
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


def streamlines_to_pts_dir_norm(streamlines, asymmetric=False):
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
    segments = streamlines_to_segments(streamlines)
    seg_mid = get_segments_mid_pts_positions(segments)
    seg_dir, seg_norm = get_segments_dir_and_norm(segments, seg_mid, asymmetric)
    return seg_mid, seg_dir, seg_norm


def get_segments_mid_pts_positions(segments):
    return 0.5 * (segments[0] + segments[1])


def get_segments_vectors(segments):
    return segments[1] - segments[0]


def get_segments_dir_and_norm(segments, seg_mid=None, asymmetric=False):
    if asymmetric:
        return get_vectors_dir_and_norm_relative_to_center(get_segments_vectors(segments), seg_mid)
    return get_vectors_dir_and_norm(get_segments_vectors(segments))


def get_vectors_dir_and_norm(vectors):
    vectors_norm = compute_vectors_norm(vectors)
    vectors_dir = vectors / vectors_norm.reshape((-1, 1))
    return vectors_dir, vectors_norm

def get_vectors_dir_and_norm_relative_to_center(vectors, seg_mid_pts):
    """ Evaluates vectors direction and norm by taking into account the
        orientation and position of segments in relation to the center 
        of voxel
    """
    vectors_norm = compute_vectors_norm(vectors)
    vectors_dir = vectors / vectors_norm.reshape((-1, 1))

    # we create an array of voxel centers for each of our points
    vox_centers = seg_mid_pts.astype(np.int) + 0.5

    # directions to center of voxel for each segment
    dir_to_center = vox_centers - seg_mid_pts

    # compute dot product between direction of vectors and direction to center
    dots = np.einsum('ij,ij->i',vectors_dir, dir_to_center)
    dots = np.reshape(dots, (dots.size, 1))

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
    _, dir_sphere_id = sphere_kdtree.query(vectors, k=1, n_jobs=-1)
    return dir_sphere_id


# Generic Functions (vector norm)
def compute_vectors_norm(vectors):
    return norm(vectors, ord=2, axis=-1)


def normalize_vectors(vectors):
    return p_normalize_vectors(vectors, 2)


def p_normalize_vectors(vectors, p):
    return vectors / norm(vectors, ord=p, axis=-1, keepdims=True)
