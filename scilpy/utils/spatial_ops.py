# -*- coding: utf-8 -*-

import numpy as np


def parallel_transport_streamline(streamline, nb_streamlines, radius, rng=None):
    """ Generate new streamlines by parallel transport of the input
    streamline. See [0] and [1] for more details.

    [0]: Hanson, A.J., & Ma, H. (1995). Parallel Transport Approach to 
        Curve Framing. # noqa E501
    [1]: TD Essentials: Parallel Transport.
        https://www.youtube.com/watch?v=5LedteSEgOE

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
    if rng is None:
        rng = np.random.default_rng(0)

    # Compute the tangent at each point of the streamline
    T = np.gradient(streamline, axis=0)
    # Normalize the tangents
    T = T / np.linalg.norm(T, axis=1)[:, None]

    # Placeholder for the normal vector at each point
    V = np.zeros_like(T)
    # Set the normal vector at the first point to be [0, 1, 0]
    # (arbitrary choice)
    V[0] = np.roll(streamline[0] - streamline[1], 1)
    V[0] = V[0] / np.linalg.norm(V[0])
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
        rand_v = rng.uniform(-1, 1)
        rand_w = rng.uniform(-1, 1)

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

        new_s = streamline + (rng.uniform(0, 1) * VW / norm) * radius
        new_streamlines.append(new_s)

    return new_streamlines
