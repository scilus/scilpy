# -*- coding: utf-8 -*-
import numpy as np
from dipy.reconst.shm import order_from_ncoef, sph_harm_ind_list


def compute_rish(sh_img):
    """Compute the RISH (Rotationally Invariant Spherical Harmonics) features
    of the SH signal [1]. Each RISH feature map is the total energy of its
    associated order. Mathematically, it is the sum of the squared SH
    coefficients of the SH order.

    Parameters
    ----------
    sh_img : nib.Nifti1Image object
        Image of the SH coefficients

    Returns
    -------
    rish_image : np.ndarray with shape (x,y,z,n_orders)
        The RISH features of the input SH, with one channel per SH order.

    References
    ----------
    [1] Mirzaalian, Hengameh, et al. "Harmonizing diffusion MRI data across
    multiple sites and scanners." MICCAI 2015.
    https://scholar.harvard.edu/files/hengameh/files/miccai2015.pdf
    """
    # Guess SH order
    sh_order = order_from_ncoef(sh_img.shape[-1], full_basis=False)

    # Get degree / order for all indices
    degree_ids, order_ids = sph_harm_ind_list(sh_order, full_basis=False)

    # Load data
    sh_data = sh_img.get_fdata(dtype=np.float32)

    # Get number of indices per order (e.g. for order 6 : [1,5,9,13])
    n_indices_per_order = np.bincount(order_ids)[::2]

    # Get start index of each order (e.g. for order 6 : [0,1,6,15])
    order_positions = np.concatenate([[0], np.cumsum(n_indices_per_order)])[:-1]

    # Get paired indices for np.add.reduceat, specifying where to reduce.
    # The last index is omitted, it is automatically replaced by len(array)-1
    # (e.g. for order 6 : [0,1, 1,6, 6,15, 15,])
    reduce_indices = np.repeat(order_positions, 2)[1:]

    # Compute the sum of squared coefficients using numpy's `reduceat`
    squared_sh = np.square(sh_data)
    rish = np.add.reduceat(squared_sh, reduce_indices, axis=-1)[..., ::2]

    return rish
