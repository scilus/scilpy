# -*- coding: utf-8 -*-

from dipy.align.imaffine import (AffineMap,
                                 AffineRegistration,
                                 MutualInformationMetric,
                                 transform_centers_of_mass)
from dipy.align.transforms import (AffineTransform3D,
                                   RigidTransform3D)
from dipy.io.utils import get_reference_info
import nibabel as nib
import numpy as np


def transform_anatomy(transfo, reference, moving, filename_to_save,
                      interp='linear'):
    """
    Apply transformation to an image using Dipy's tool

    Parameters
    ----------
    transfo: numpy.ndarray
        Transformation matrix to be applied
    reference: str
        Filename of the reference image (target)
    moving: str
        Filename of the moving image
    filename_to_save: str
        Filename of the output image
    interp : string, either 'linear' or 'nearest'
        the type of interpolation to be used, either 'linear'
        (for k-linear interpolation) or 'nearest' for nearest neighbor
    """
    grid2world, dim, _, _ = get_reference_info(reference)
    static_data = nib.load(reference).get_fdata(dtype=np.float32)

    nib_file = nib.load(moving)
    moving_data = nib_file.get_fdata(dtype=np.float32)
    moving_affine = nib_file.affine

    if moving_data.ndim == 3 and isinstance(moving_data[0, 0, 0],
                                            np.ScalarType):
        orig_type = moving_data.dtype
        affine_map = AffineMap(np.linalg.inv(transfo),
                               dim, grid2world,
                               moving_data.shape, moving_affine)
        resampled = affine_map.transform(moving_data.astype(np.float64),
                                         interp=interp)
        nib.save(nib.Nifti1Image(resampled.astype(orig_type), grid2world),
                 filename_to_save)
    elif len(moving_data[0, 0, 0]) > 1:
        if isinstance(moving_data[0, 0, 0], np.void):
            raise ValueError('Does not support TrackVis RGB')

        affine_map = AffineMap(np.linalg.inv(transfo),
                               dim[0:3], grid2world,
                               moving_data.shape[0:3], moving_affine)

        orig_type = moving_data.dtype
        resampled = transform_dwi(affine_map, static_data, moving_data,
                                  interp=interp)
        nib.save(nib.Nifti1Image(resampled.astype(orig_type), grid2world),
                 filename_to_save)
    else:
        raise ValueError('Does not support this dataset (shape, type, etc)')


def transform_dwi(reg_obj, static, dwi, interp='linear'):
    """
    Iteratively apply transformation to 4D image using Dipy's tool

    Parameters
    ----------
    reg_obj: AffineMap
        Registration object from Dipy returned by AffineMap
    static: numpy.ndarray
        Target image data
    dwi: numpy.ndarray
        4D numpy array containing a scalar in each voxel (moving image data)
    interp : string, either 'linear' or 'nearest'
        the type of interpolation to be used, either 'linear'
        (for k-linear interpolation) or 'nearest' for nearest neighbor
    """
    trans_dwi = np.zeros(static.shape + (dwi.shape[3],), dtype=dwi.dtype)
    for i in range(dwi.shape[3]):
        trans_dwi[..., i] = reg_obj.transform(dwi[..., i], interp=interp)

    return trans_dwi


def register_image(static, static_grid2world, moving, moving_grid2world,
                   transformation_type='affine', dwi=None):
    if transformation_type not in ['rigid', 'affine']:
        raise ValueError('Transformation type not available in Dipy')

    # Set all parameters for registration
    nbins = 32
    params0 = None
    sampling_prop = None
    level_iters = [50, 25, 5]
    sigmas = [8.0, 4.0, 2.0]
    factors = [8, 4, 2]
    metric = MutualInformationMetric(nbins, sampling_prop)
    reg_obj = AffineRegistration(metric=metric, level_iters=level_iters,
                                 sigmas=sigmas, factors=factors, verbosity=0)

    # First, align the center of mass of both volume
    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)
    # Then, rigid transformation (translation + rotation)
    transform = RigidTransform3D()
    rigid = reg_obj.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=c_of_mass.affine)

    if transformation_type == 'affine':
        # Finally, affine transformation (translation + rotation + scaling)
        transform = AffineTransform3D()
        affine = reg_obj.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=rigid.affine)

        mapper = affine
        transformation = affine.affine
    else:
        mapper = rigid
        transformation = rigid.affine

    if dwi is not None:
        trans_dwi = transform_dwi(mapper, static, dwi)
        return trans_dwi, transformation
    else:
        return mapper.transform(moving), transformation
