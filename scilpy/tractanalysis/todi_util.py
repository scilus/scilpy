# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm
from scipy.spatial.ckdtree import cKDTree


def streamlines_to_segments(streamlines):
    vts_0_list = []
    vts_1_list = []
    for streamline in streamlines:
        vts_0_list.append(streamline[:-1])
        vts_1_list.append(streamline[1:])

    segments = np.stack((np.vstack(vts_0_list), np.vstack(vts_1_list)), axis=0)
    return segments


def streamlines_to_endpoints(streamlines):
    endpoints = np.zeros((2, len(streamlines), 3))
    for i, streamline in enumerate(streamlines):
        endpoints[0, i] = streamline[0]
        endpoints[1, i] = streamline[-1]

    return endpoints


def streamlines_to_pts_dir_norm(streamlines):
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
    # sphere_vertices must be normed (or all with equal norm)
    sphere_kdtree = cKDTree(sphere_vertices)
    _, dir_sphere_id = sphere_kdtree.query(vectors, k=1, n_jobs=-1)
    return dir_sphere_id


# Generic Functions (vector norm)
def compute_vectors_norm(vectors):
    return norm(vectors, ord=2, axis=-1)


def normalize_vectors(vectors):
    return vectors / norm(vectors, ord=2, axis=-1, keepdims=True)


def p_normalize_vectors(vectors, p):
    return vectors / norm(vectors, ord=p, axis=-1, keepdims=True)
