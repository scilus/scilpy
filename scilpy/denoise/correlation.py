# -*- coding: utf-8 -*-


def correlate_sf_from_sh(in_sh, sphere, get_filter_func, sigma_range=None):
    """
    Correlation operation on SF amplitudes from SH coefficients array.

    Parameters
    ----------
    in_sh: ndarray (X, Y, Z, n_coeffs)
        SH coefficients image to correlate.
    sphere: dipy.sphere
        Sphere used for SH to SF projection.
    get_filter_func: Callable(direction)
        Callback taking a sphere direction as input
        and returning filter weights for this direction.
    sigma_range: float or None, optional
        If given, a range kernel is applied on-the-fly for edges preservation.

    Returns
    -------
    out_sh: ndarray (X, Y, Z, n_coeffs_full)
        Output SH image saved in full SH basis to preserve asymmetries.
    """
    pass
