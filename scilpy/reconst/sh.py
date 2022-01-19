# -*- coding: utf-8 -*-
import numpy as np
from dipy.reconst.shm import order_from_ncoef, sph_harm_ind_list


def compute_rish(sh, mask=None, full_basis=False):
    """Compute the RISH (Rotationally Invariant Spherical Harmonics) features
    of the SH signal [1]. Each RISH feature map is the total energy of its
    associated order. Mathematically, it is the sum of the squared SH
    coefficients of the SH order.

    Parameters
    ----------
    sh : np.ndarray object
        Array of the SH coefficients
    mask: np.ndarray object, optional
        Binary mask. Only data inside the mask will be used for computation.
    full_basis: bool, optional
        True when coefficients are for a full SH basis.

    Returns
    -------
    rish : np.ndarray with shape (x,y,z,n_orders)
        The RISH features of the input SH, with one channel per SH order.
    orders : list(int)
        The SH order of each RISH feature in the last channel of `rish`.

    References
    ----------
    [1] Mirzaalian, Hengameh, et al. "Harmonizing diffusion MRI data across
        multiple sites and scanners." MICCAI 2015.
        https://scholar.harvard.edu/files/hengameh/files/miccai2015.pdf
    """
    # Guess SH order
    sh_order = order_from_ncoef(sh.shape[-1], full_basis=full_basis)

    # Get degree / order for all indices
    degree_ids, order_ids = sph_harm_ind_list(sh_order, full_basis=full_basis)

    # Apply mask to input
    if mask is not None:
        sh = sh * mask[..., None]

    # Get number of indices per order (e.g. for order 6, sym. : [1,5,9,13])
    step = 1 if full_basis else 2
    n_indices_per_order = np.bincount(order_ids)[::step]

    # Get start index of each order (e.g. for order 6 : [0,1,6,15])
    order_positions = np.concatenate([[0], np.cumsum(n_indices_per_order)])[:-1]

    # Get paired indices for np.add.reduceat, specifying where to reduce.
    # The last index is omitted, it is automatically replaced by len(array)-1
    # (e.g. for order 6 : [0,1, 1,6, 6,15, 15,])
    reduce_indices = np.repeat(order_positions, 2)[1:]

    # Compute the sum of squared coefficients using numpy's `reduceat`
    squared_sh = np.square(sh)
    rish = np.add.reduceat(squared_sh, reduce_indices, axis=-1)[..., ::2]

    # Apply mask
    if mask is not None:
        rish *= mask[..., None]

    orders = sorted(np.unique(order_ids))

    return rish, orders
