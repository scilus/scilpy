# -*- coding: utf-8 -*-
# THIS FILE IS TEMPORARY. THIS FILE WILL BE DELETED AFTER THE PR IN DIPY

from multiprocessing import Pool, cpu_count
import warnings

import numpy as np
from scipy.ndimage import affine_transform


def _affine_transform(kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*scipy.*18.*",
                                category=UserWarning)
        return affine_transform(**kwargs)


def reslice(data, affine, zooms, new_zooms, order=1, mode='constant', cval=0,
            num_processes=1):
    """Reslice data with new voxel resolution defined by ``new_zooms``

    Parameters
    ----------
    data : array, shape (I,J,K) or (I,J,K,N)
        3d volume or 4d volume with datasets
    affine : array, shape (4,4)
        mapping from voxel coordinates to world coordinates
    zooms : tuple, shape (3,)
        voxel size for (i,j,k) dimensions
    new_zooms : tuple, shape (3,)
        new voxel size for (i,j,k) after resampling
    order : int, from 0 to 5
        order of interpolation for resampling/reslicing,
        0 nearest interpolation, 1 trilinear etc..
        if you don't want any smoothing 0 is the option you need.
    mode : string ('constant', 'nearest', 'reflect' or 'wrap')
        Points outside the boundaries of the input are filled according
        to the given mode.
    cval : float
        Value used for points outside the boundaries of the input if
        mode='constant'.
    num_processes : int
        Split the calculation to a pool of children processes. This only
        applies to 4D `data` arrays. If a positive integer then it defines
        the size of the multiprocessing pool that will be used. If 0, then
        the size of the pool will equal the number of cores available.

    Returns
    -------
    data2 : array, shape (I,J,K) or (I,J,K,N)
        datasets resampled into isotropic voxel size
    affine2 : array, shape (4,4)
        new affine for the resampled image

    Examples
    --------
    >>> from dipy.io.image import load_nifti
    >>> from dipy.align.reslice import reslice
    >>> from dipy.data import get_fnames
    >>> f_name = get_fnames('aniso_vox')
    >>> data, affine, zooms = load_nifti(f_name, return_voxsize=True)
    >>> data.shape == (58, 58, 24)
    True
    >>> zooms
    (4.0, 4.0, 5.0)
    >>> new_zooms = (3.,3.,3.)
    >>> new_zooms
    (3.0, 3.0, 3.0)
    >>> data2, affine2 = reslice(data, affine, zooms, new_zooms)
    >>> data2.shape == (77, 77, 40)
    True
    """
    # We are suppressing warnings emitted by scipy >= 0.18,
    # described in https://github.com/dipy/dipy/issues/1107.
    # These warnings are not relevant to us, as long as our offset
    # input to scipy's affine_transform is [0, 0, 0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*scipy.*18.*",
                                category=UserWarning)
        new_zooms = np.array(new_zooms, dtype='f8')
        zooms = np.array(zooms, dtype='f8')
        R = new_zooms / zooms

        original_extent = []
        offset = []
        axes = []
        Rx = np.eye(4)
        Rx[:3, :3] = np.diag(1/zooms)
        affine = np.dot(affine, Rx)
        for j in range(3):
            original_extent.append(data.shape[j] * zooms[j])
            axes.append(
                np.round(data.shape[j] * zooms[j] / new_zooms[j] - 0.0001))
            offset.append(0.5 * ((new_zooms[j] - zooms[j]) + (
                                original_extent[j] - (axes[j] * new_zooms[j]))) / zooms[j])
            for i in range(3):
                affine[i, 3] += 0.5 * ((new_zooms[j] - zooms[j]) + (
                            original_extent[j] - (axes[j] * new_zooms[j]))) * \
                                 affine[i, j]
        Rx = np.eye(4)
        Rx[:3, :3] = np.diag(new_zooms)
        affine2 = np.dot(affine, Rx)

        new_shape = zooms / new_zooms * np.array(data.shape[:3])
        new_shape = tuple(np.round(new_shape).astype('i8'))
        kwargs = {'matrix': R, 'output_shape': new_shape, 'order': order,
                  'mode': mode, 'cval': cval, 'offset': offset}
        if data.ndim == 3:
            data2 = affine_transform(input=data, **kwargs)
        if data.ndim == 4:
            data2 = np.zeros(new_shape+(data.shape[-1],), data.dtype)
            if not num_processes:
                num_processes = cpu_count()
            if num_processes < 2:
                for i in range(data.shape[-1]):
                    affine_transform(input=data[..., i], output=data2[..., i],
                                     **kwargs)
            else:
                params = []
                for i in range(data.shape[-1]):
                    _kwargs = {'input': data[..., i]}
                    _kwargs.update(kwargs)
                    params.append(_kwargs)
                pool = Pool(num_processes)
                for i, res in enumerate(pool.imap(_affine_transform, params)):
                    data2[..., i] = res
                pool.close()

    return data2, affine2
