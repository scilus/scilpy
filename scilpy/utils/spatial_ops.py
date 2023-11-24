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


def project_on_plane(vector, u, w):
    """Project a vector onto the plane defined by vectors u and w."""
    # Calculate projections onto u and w
    proj_u = np.dot(vector, u) * u
    proj_w = np.dot(vector, w) * w

    # Combine projections
    projection = proj_u + proj_w

    # Normalize and scale to match the original vector's norm
    norm_vector = np.linalg.norm(vector)
    normalized_projection = normalize_vector(projection)
    scaled_projection = normalized_projection * norm_vector

    return scaled_projection


def parallel_transport_streamline(streamline, nb_streamlines, radius):
    """ Generate new streamlines by parallel transport of the input
    streamline. See [0] and [1] for more details.

    [0]: Hanson, A.J., & Ma, H. (1995). Parallel Transport Approach to Curve Framing. # noqa E501
    [1]: TD Essentials: Parallel Transport. https://www.youtube.com/watch?v=5LedteSEgOE

    Parameters
    ----------
    streamline: ndarray (N, 3)
        The streamline to transport.
    nb_streamlines: int
        The number of streamlines to generate.

    Returns
    -------
    new_streamlines: list of ndarray (N, 3)
        The generated streamlines.
    """

    def r(vec, theta):
        """ Rotation matrix around a 3D vector by an angle theta.
        From https://stackoverflow.com/questions/6802577/rotation-of-3d-vector

        TODO?: Put this somewhere else.

        Parameters
        ----------
        vec: ndarray (3,)
            The vector to rotate around.
        theta: float
            The angle of rotation in radians.

        Returns
        -------
        rot: ndarray (3, 3)
            The rotation matrix.
        """

        vec = vec / np.linalg.norm(vec)
        x, y, z = vec
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c + x**2 * (1 - c),
                          x * y * (1 - c) - z * s,
                          x * z * (1 - c) + y * s],
                         [y * x * (1 - c) + z * s,
                          c + y**2 * (1 - c),
                          y * z * (1 - c) - x * s],
                         [z * x * (1 - c) - y * s,
                          z * y * (1 - c) + x * s,
                          c + z**2 * (1 - c)]])

    # Compute the tangent at each point of the streamline
    T = np.gradient(streamline, axis=0)
    # Normalize the tangents
    T = T / np.linalg.norm(T, axis=1)[:, None]

    # Placeholder for the normal vector at each point
    V = np.zeros_like(T)
    # Set the normal vector at the first point to be [0, 1, 0]
    # (arbitrary choice)
    V[0] = np.random.rand(3)

    # For each point
    for i in range(0, T.shape[0]-1):
        # Compute the torsion vector
        B = np.cross(T[i], T[i+1])
        # If the torsion vector is 0, the normal vector does not change
        if np.linalg.norm(B) < 1e-3:
            V[i+1] = V[i]
        # Else, the normal vector is rotated around the torsion vector by
        # the torsion.
        else:
            B = B / np.linalg.norm(B)
            theta = np.arccos(np.dot(T[i], T[i+1]))
            # Rotate the vector V[i] around the vector B by theta
            # radians.
            V[i+1] = np.dot(r(B, theta), V[i])

    # Compute the binormal vector at each point
    W = np.cross(T, V, axis=1)

    # Generate the new streamlines
    # TODO?: This could easily be optimized to avoid the for loop, we have to
    # see if this becomes a bottleneck.
    new_streamlines = []
    for i in range(nb_streamlines):
        # Get a random number between -1 and 1
        rand_v = (np.random.rand() * 2 - 1)
        rand_w = (np.random.rand() * 2 - 1)
        # Compute the norm of the "displacement"
        norm = np.sqrt(rand_v**2 + rand_w**2)
        # Displace the normal and binormal vectors by a random amount
        V_mod = V * rand_v
        W_mod = W * rand_w
        # Compute the displacement vector
        VW = (V_mod + W_mod)
        # Displace the streamline around the original one following the
        # parallel frame. Make sure to normalize the displacement vector
        # so that the new streamline is in a circle around the original one.

        new_s = streamline + (np.random.rand() * VW / norm) * radius
        new_streamlines.append(new_s)

    return new_streamlines
