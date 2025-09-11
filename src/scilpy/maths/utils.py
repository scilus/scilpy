# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import svd


def _fit_circle_2d(x, y, dist_w):
    """
    Least square for circle fitting in 2D.

    Parameters
    ----------
    x: np.ndarray
        x coordinates
    y: np.ndarray
        y coordinates
    dist_w: np.ndarray
        Radius. Allows for re-weighting of points.

    Returns
    -------
    x_c: np.ndarray
        Fitted circle coordinates (x_coordinates)
    y_c: np.ndarray
        Fitted circle coordinates (y_coordinates)
    r: float
        The radius of the circle.
    """
    if dist_w is None:
        dist_w = np.ones(len(x))

    # Fit a circle in 2D using least-squares
    A = np.array([x, y, dist_w]).T
    b = x**2 + y**2
    params = np.linalg.lstsq(A, b, rcond=None)[0]

    # Get circle parameters from solution
    x_c = params[0]/2
    y_c = params[1]/2
    r = np.sqrt(params[2] + x_c**2 + y_c**2)

    return x_c, y_c, r

def _rodrigues_rot(pts, n0, n1):
    """
    Rodrigues rotation [1]
    - Rotate given points based on a starting and ending vector
    - Axis k and angle of rotation theta given by vectors n0,n1
    P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))

    Parameters
    ----------
    pts: np.ndarray
        The coordinates of the points.
    n0: np.ndarray
        The starting vector.
    n1: np.ndarray
        The ending vector.

    Returns
    -------
    pts_rot: np.ndarray
        The rotated coordinates of the points.

    References
    ----------
    [1] https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
    """

    # If P is only 1d array (coords of single point), fix it to be matrix
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]

    # Get vector of rotation k and angle theta
    n0 /= np.linalg.norm(n0)
    n1 /= np.linalg.norm(n1)
    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(n0, n1))

    # Compute rotated points
    pts_rot = np.zeros((len(pts), 3))
    for i in range(len(pts)):
        pts_rot[i] = pts[i] * np.cos(theta) + np.cross(k, pts[i]) * \
                   np.sin(theta) + k * np.dot(k, pts[i]) * (1 - np.cos(theta))

    return pts_rot


def fit_circle_planar(pts, dist_w):
    """
    Fitting a plane by SVD for the mean-centered data.

    Parameters
    ----------
    pts: np.ndarray
        The coordinates.
    dist_w: str
        One of ['lin_up', 'lin_down', 'exp', 'inv', 'log'].

    Returns
    -------
    pts_recentered: np.ndarray
        The fitted coordinates.
    radius: float
        The radius of the circle.
    """
    pts_mean = pts.mean(axis=0)
    pts_centered = pts - pts_mean

    _, _, V = svd(pts_centered, full_matrices=False)
    normal = V[2, :]

    # Project points to coords X-Y in 2D plane
    pts_xy = _rodrigues_rot(pts_centered, normal, [0, 0, 1])

    # Fit circle in new 2D coords
    dist = np.linalg.norm(pts_centered, axis=1)
    if dist_w == 'lin_up':
        dist /= np.max(dist)
    elif dist_w == 'lin_down':
        dist /= np.max(dist)
        dist = 1 - dist
    elif dist_w == 'exp':
        dist /= np.max(dist)
        dist = np.exp(dist)
    elif dist_w == 'inv':
        dist /= np.max(dist)
        dist = 1 / dist
    elif dist_w == 'log':
        dist /= np.max(dist)
        dist = np.log(dist+1)
    elif dist is not None:
        raise ValueError("Invalid dist_w. Should be one of 'lin_up', "
                         "'lin_down', 'exp', 'inv', 'log', or None, but "
                         "got {}".format(dist_w))

    x_c, y_c, radius = _fit_circle_2d(pts_xy[:, 0], pts_xy[:, 1], dist)

    # Transform circle center back to 3D coords
    pts_recentered = _rodrigues_rot(np.array([x_c, y_c, 0]),
                                    [0, 0, 1], normal) + pts_mean

    return pts_recentered, radius
