# -*- coding: utf-8 -*-

import logging

from dipy.data import get_sphere
import numpy as np

from scilpy.reconst.utils import find_order_from_nb_coeff, get_b_matrix


def get_ventricles_max_fodf(data, fa, md, zoom, args):
    """
    Compute mean maximal fodf value in ventricules. Given
    heuristics thresholds on FA and MD values, finds the
    voxels of the ventricules or CSF and computes a mean
    fODF value. This is described in
    Dell'Acqua et al HBM 2013.

    Parameters
    ----------
    data: ndarray (x, y, z, ncoeffs)
         Input fODF file in spherical harmonics coefficients.
    fa: ndarray (x, y, z)
         FA (Fractional Anisotropy) volume from DTI
    md: ndarray (x, y, z)
         MD (Mean Diffusivity) volume from DTI
    vol: int > 0
         Maximum Nnumber of voxels used to compute the mean.
         1000 works well at 2x2x2 = 8 mm3

    Returns
    -------
    mean, mask: int, ndarray (x, y, z)
         Mean maximum fODF value and mask of voxels used
    """

    order = find_order_from_nb_coeff(data)
    sphere = get_sphere('repulsion100')
    b_matrix = get_b_matrix(order, sphere, args.sh_basis)
    sum_of_max = 0
    count = 0

    mask = np.zeros(data.shape[:-1])

    if np.min(data.shape[:-1]) > 40:
        step = 20
    else:
        if np.min(data.shape[:-1]) > 20:
            step = 10
        else:
            step = 5

    # 1000 works well at 2x2x2 = 8 mm3
    # Hence, we multiply by the volume of a voxel
    vol = (zoom[0] * zoom[1] * zoom[2])
    if vol != 0:
        max_number_of_voxels = 1000 * 8 // vol
    else:
        max_number_of_voxels = 1000

    # In the case of 2D-like data (3D data with one dimension size of 1), or
    # a small 3D dataset, the full range of data is scanned.
    if args.small_dims:
        all_i = list(range(0, data.shape[0]))
        all_j = list(range(0, data.shape[1]))
        all_k = list(range(0, data.shape[2]))
    # In the case of a normal 3D dataset, a window is created in the middle of
    # the image to capture the ventricules. No need to scan the whole image.
    else:
        all_i = list(range(int(data.shape[0]/2) - step,
                           int(data.shape[0]/2) + step))
        all_j = list(range(int(data.shape[1]/2) - step,
                           int(data.shape[1]/2) + step))
        all_k = list(range(int(data.shape[2]/2) - step,
                           int(data.shape[2]/2) + step))
    for i in all_i:
        for j in all_j:
            for k in all_k:
                if count > max_number_of_voxels - 1:
                    continue
                if fa[i, j, k] < args.fa_threshold \
                        and md[i, j, k] > args.md_threshold:
                    sf = np.dot(data[i, j, k], b_matrix.T)
                    sum_of_max += sf.max()
                    count += 1
                    mask[i, j, k] = 1

    logging.debug('Number of voxels detected: {}'.format(count))
    if count == 0:
        logging.warning('No voxels found for evaluation! Change your fa '
                        'and/or md thresholds')
        return 0, mask

    logging.debug('Average max fodf value: {}'.format(sum_of_max / count))
    return sum_of_max / count, mask
