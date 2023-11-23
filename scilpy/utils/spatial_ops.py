# -*- coding: utf-8 -*-

import numpy as np


def normalize_vector(v):
    """Normalize a 3D vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def find_perpendicular_vectors(v):
    """Find two perpendicular unit vectors for a given normalized 3D vector."""
    if v[0] == 0 and v[1] == 0:
        if v[2] == 0:
            # v is a zero vector, can't find perpendicular vectors
            return None, None
        # v is along the z-axis, choose x and y axes as perpendicular vectors
        return np.array([1, 0, 0]), np.array([0, 1, 0])

    # General case, find one vector perpendicular to v and z-axis
    u = np.cross(v, [0, 0, 1])
    u = normalize_vector(u)

    # Find another vector perpendicular to both v and u
    w = np.cross(v, u)
    w = normalize_vector(w)

    return u, w


def sample_points_on_circle(center, radius, normal, num_points):
    """Sample points on a circle perpendicular to the given normal vector."""
    normal = normalize_vector(normal)
    u, w = find_perpendicular_vectors(normal)

    if u is None or w is None:
        return None

    points = []
    value = np.random.random(num_points) * 2 * np.pi
    for theta in value:
        point = center + radius * (np.cos(theta) * u + np.sin(theta) * w)
        points.append(point)

    return np.array(points)
