# -*- coding: utf-8 -*-

import logging

from scilpy.image.reslice import reslice  # Don't use Dipy's reslice. Buggy.
import nibabel as nib
import numpy as np


def _interp_code_to_order(interp_code):
    orders = {'nn': 0, 'lin': 1, 'quad': 2, 'cubic': 3}
    return orders[interp_code]


def resample_volume(img, ref=None, res=None, iso_min=False, zoom=None,
                    interp='lin', enforce_dimensions=False):
    """
    Function to resample a dataset to match the resolution of another
    reference dataset or to the resolution specified as in argument.
    One of the following options must be chosen: ref, res or iso_min.

    Parameters
    ----------
    img: nib.Nifti1Image
        Image to resample.
    ref: nib.Nifti1Image
        Reference volume to resample to. This method is used only if ref is not
        None. (default: None)
    res: tuple, shape (3,) or int, optional
        Resolution to resample to. If the value it is set to is Y, it will
        resample to an isotropic resolution of Y x Y x Y. This method is used
        only if res is not None. (default: None)
    iso_min: bool, optional
        If true, resample the volume to R x R x R with R being the smallest
        current voxel dimension. If false, this method is not used.
    zoom: tuple, shape (3,) or float, optional
        Set the zoom property of the image at the value specified.
    interp: str, optional
        Interpolation mode. 'nn' = nearest neighbour, 'lin' = linear,
        'quad' = quadratic, 'cubic' = cubic. (Default: linear)
    enforce_dimensions: bool, optional
        If True, enforce the reference volume dimension (only if res is not
        None). (Default = False)

    Returns
    -------
    resampled_image: nib.Nifti1Image
        Resampled image.
    """
    data = np.asanyarray(img.dataobj)
    original_res = data.shape
    affine = img.affine
    original_zooms = img.header.get_zooms()[:3]

    if ref is not None:
        if iso_min or zoom or res:
            raise ValueError('Please only provide one option amongst ref, res '
                             ', zoom or iso_min.')
        ref_img = nib.load(ref)
        new_zooms = ref_img.header.get_zooms()[:3]
    elif res is not None:
        if iso_min or zoom:
            raise ValueError('Please only provide one option amongst ref, res '
                             ', zoom or iso_min.')
        if len(res) == 1:
            res = res * 3
        new_zooms = tuple((o / r) * z for o, r,
                          z in zip(original_res, res, original_zooms))

    elif iso_min:
        if zoom:
            raise ValueError('Please only provide one option amongst ref, res '
                             ', zoom or iso_min.')
        min_zoom = min(original_zooms)
        new_zooms = (min_zoom, min_zoom, min_zoom)
    elif zoom:
        new_zooms = zoom
        if len(zoom) == 1:
            new_zooms = zoom * 3
    else:
        raise ValueError("You must choose the resampling method. Either with"
                         "a reference volume, or a chosen isometric resolution"
                         ", or an isometric resampling to the smallest current"
                         " voxel dimension!")

    interp_choices = ['nn', 'lin', 'quad', 'cubic']
    if interp not in interp_choices:
        raise ValueError("interp must be one of 'nn', 'lin', 'quad', 'cubic'.")

    logging.debug('Data shape: %s', data.shape)
    logging.debug('Data affine: %s', affine)
    logging.debug('Data affine setup: %s', nib.aff2axcodes(affine))
    logging.debug('Resampling data to %s with mode %s', new_zooms, interp)

    data2, affine2 = reslice(data, affine, original_zooms, new_zooms,
                             _interp_code_to_order(interp))

    logging.debug('Resampled data shape: %s', data2.shape)
    logging.debug('Resampled data affine: %s', affine2)
    logging.debug('Resampled data affine setup: %s', nib.aff2axcodes(affine2))

    if enforce_dimensions:
        if ref is None:
            raise ValueError('enforce_dimensions can only be used with the ref'
                             'method.')
        else:
            computed_dims = data2.shape
            ref_dims = ref_img.shape[:3]
            if computed_dims != ref_dims:
                fix_dim_volume = np.zeros(ref_dims)
                x_dim = min(computed_dims[0], ref_dims[0])
                y_dim = min(computed_dims[1], ref_dims[1])
                z_dim = min(computed_dims[2], ref_dims[2])

                fix_dim_volume[:x_dim, :y_dim, :z_dim] = \
                    data2[:x_dim, :y_dim, :z_dim]
                data2 = fix_dim_volume

    return nib.Nifti1Image(data2.astype(data.dtype), affine2)
