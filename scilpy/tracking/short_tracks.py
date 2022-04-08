# -*- coding: utf-8 -*-

def track_short_tracks(in_odf, in_seed, in_mask, step_size=0.5,
                       min_length=10., max_length=20.,
                       theta=20.0, sh_order=8,
                       sh_basis='descoteaux07'):
    """
    Track short tracks.

    Parameters
    ----------
    in_odf : ndarray
        Spherical harmonics volume. Ex: ODF or fODF.
    in_mask : ndarray
        Tracking mask. Tracking stops outside the mask. Seeding is uniform
        inside the mask.
    step_size : float, optional
        Step size in mm. [0.5]
    min_length : float, optional
        Minimum length of a streamline in mm. [10.]
    max_length : float, optional
        Maximum length of a streamline in mm. [20.]
    theta : float, optional
        Maximum angle (degrees) between 2 steps.
    sh_order : int, optional
        Spherical harmonics order. [8]
    sh_basis : str, optional
        Spherical harmonics basis. ['descoteaux07']

    Returns
    -------
    streamlines: list
        List of short-tracks.
    """
