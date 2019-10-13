# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm
from scipy.spatial.ckdtree import cKDTree


def streamlines_to_segments(streamlines):
    """Split streamlines into its segments
    :param streamlines: list, streamlines with short sampling
    :return numpy.ndarray (nbr of segments, 2)
    """
    vts_0_list = []
    vts_1_list = []
    for streamline in streamlines:
        vts_0_list.append(streamline[:-1])
        vts_1_list.append(streamline[1:])

    segments = np.stack((np.vstack(vts_0_list), np.vstack(vts_1_list)), axis=0)
    return segments


def streamlines_to_endpoints(streamlines):
    """Equivalent to resampling to 2 points (first and last)
    :param streamlines: list, streamlines with short sampling
    :return numpy.ndarray (nbr of streamlines, 2)
    """
    endpoints = np.zeros((2, len(streamlines), 3))
    for i, streamline in enumerate(streamlines):
        endpoints[0, i] = streamline[0]
        endpoints[1, i] = streamline[-1]

    return endpoints


def streamlines_to_pts_dir_norm(streamlines):
    """Evaluate each segment attributes
    :param streamlines: list, streamlines with short sampling
    :return tuple, 3 numpy.ndarray
        seg_mid, XYZ coordinates of the center of each segment
        seg_dir, XYZ orientation of each segment
        seg_norm, float value representing the norm of each segment
    """
    segments = streamlines_to_segments(streamlines)
    seg_mid = get_segments_mid_pts_positions(segments)
    seg_dir, seg_norm = get_segments_dir_and_norm(segments)
    return seg_mid, seg_dir, seg_norm


def get_segments_mid_pts_positions(segments):
    return 0.5 * (segments[0] + segments[1])


def get_segments_vectors(segments):
    return segments[1] - segments[0]


def get_segments_dir_and_norm(segments):
    return get_vectors_dir_and_norm(get_segments_vectors(segments))


def get_vectors_dir_and_norm(vectors):
    vectors_norm = compute_vectors_norm(vectors)
    vectors_dir = vectors / vectors_norm.reshape((-1, 1))
    return vectors_dir, vectors_norm


def psf_from_sphere(sphere_vertices):
    return np.abs(np.dot(sphere_vertices, sphere_vertices.T))


# Mask functions
def generate_mask_indices_1d(nb_voxel, indices_1d):
    mask_1d = np.zeros(nb_voxel, dtype=np.bool)
    mask_1d[indices_1d] = True
    return mask_1d


def get_indices_1d(volume_shape, pts):
    return np.ravel_multi_index(pts.T.astype(np.int), volume_shape)


def get_dir_to_sphere_id(vectors, sphere_vertices):
    """Find the closest vector on the sphere using a cKDT tree
        sphere_vertices must be normed (or all with equal norm)
    :param vectors: numpy.ndarray, multiple vectors to query
    :param sphere_vertices, numpy.ndarray typically from dipy sphere object
    :retun dir_sphere_id, numpy.ndarray all indices of
        the closest sphere orientation
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
