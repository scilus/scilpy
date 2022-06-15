# -*- coding: utf-8 -*-

import logging

from dipy.align.imaffine import (AffineMap,
                                 AffineRegistration,
                                 MutualInformationMetric,
                                 transform_centers_of_mass)
from dipy.align.transforms import (AffineTransform3D,
                                   RigidTransform3D)
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.utils import get_reference_info
from dipy.segment.mask import median_otsu
import nibabel as nib
import numpy as np

from scipy.ndimage.morphology import binary_dilation
from scilpy.io.image import get_data_as_mask
from scilpy.utils.bvec_bval_tools import identify_shells


def transform_anatomy(transfo, reference, moving, filename_to_save,
                      interp='linear', keep_dtype=False):
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
    keep_dtype : bool
        If True, keeps the data_type of the input moving image when saving
        the output image
    """
    grid2world, dim, _, _ = get_reference_info(reference)
    static_data = nib.load(reference).get_fdata(dtype=np.float32)

    nib_file = nib.load(moving)
    curr_type = nib_file.get_data_dtype()
    if keep_dtype:
        moving_data = np.asanyarray(nib_file.dataobj).astype(curr_type)
    else:
        moving_data = nib_file.get_fdata(dtype=np.float32)
    moving_affine = nib_file.affine

    if moving_data.ndim == 3 and isinstance(moving_data[0, 0, 0],
                                            np.ScalarType):
        orig_type = moving_data.dtype
        affine_map = AffineMap(np.linalg.inv(transfo),
                               dim, grid2world,
                               moving_data.shape, moving_affine)
        resampled = affine_map.transform(moving_data.astype(np.float64),
                                         interpolation=interp)
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
                                  interpolation=interp)
        nib.save(nib.Nifti1Image(resampled.astype(orig_type), grid2world),
                 filename_to_save)
    else:
        raise ValueError('Does not support this dataset (shape, type, etc)')


def transform_dwi(reg_obj, static, dwi, interpolation='linear'):
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
    interpolation : string, either 'linear' or 'nearest'
        the type of interpolation to be used, either 'linear'
        (for k-linear interpolation) or 'nearest' for nearest neighbor
    """
    trans_dwi = np.zeros(static.shape + (dwi.shape[3],), dtype=dwi.dtype)
    for i in range(dwi.shape[3]):
        trans_dwi[..., i] = reg_obj.transform(dwi[..., i],
                                              interpolation=interpolation)

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


def compute_snr(dwi, bval, bvec, b0_thr, mask,
                noise_mask=None, noise_map=None,
                split_shells=False,
                basename=None, verbose=False):
    """
    Compute snr

    Parameters
    ----------
    dwi: string
        Path to the dwi file
    bvec: string
        Path to the bvec file
    bval: string
        Path to the bval file
    b0_thr: int
        Threshold to define b0 minimum value
    mask: string
        Path to the mask
    noise_mask: string
        Path to the noise mask
    noise_map: string
        Path to the noise map
    basename: string
        Basename used for naming all output files

    verbose: boolean
        Set to use logging
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    img = nib.load(dwi)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    mask = get_data_as_mask(nib.load(mask), dtype=bool)
    bvals, bvecs = read_bvals_bvecs(bval, bvec)

    if split_shells:
        centroids, shell_indices = identify_shells(bvals, threshold=40.0,
                                                   roundCentroids=False,
                                                   sort=False)
        bvals = centroids[shell_indices]

    b0s_location = bvals <= b0_thr

    if not np.any(b0s_location):
        raise ValueError('You should ajust --b0_thr={} '
                         'since no b0s where find.'.format(b0_thr))

    if noise_mask is None and noise_map is None:
        b0_mask, noise_mask = median_otsu(data, vol_idx=b0s_location)

        # we inflate the mask, then invert it to recover only the noise
        noise_mask = binary_dilation(noise_mask, iterations=10).squeeze()

        # Add the upper half in order to delete the neck and shoulder
        # when inverting the mask
        noise_mask[..., :noise_mask.shape[-1]//2] = 1

        # Reverse the mask to get only noise
        noise_mask = (~noise_mask).astype('float32')

        logging.info('Number of voxels found '
                     'in noise mask : {}'.format(np.count_nonzero(noise_mask)))
        logging.info('Total number of voxel '
                     'in volume : {}'.format(np.size(noise_mask)))

        nib.save(nib.Nifti1Image(noise_mask, affine),
                 basename + '_noise_mask.nii.gz')
    elif noise_mask:
        noise_mask = get_data_as_mask(nib.load(noise_mask),
                                      dtype=bool).squeeze()
    elif noise_map:
        img_noisemap = nib.load(noise_map)
        data_noisemap = img_noisemap.get_fdata(dtype=np.float32)

    # Val = np array (mean_signal, std_noise)
    val = {0: {'bvec': [0, 0, 0], 'bval': 0, 'mean': 0, 'std': 0}}
    for idx in range(data.shape[-1]):
        val[idx] = {}
        val[idx]['bvec'] = bvecs[idx]
        val[idx]['bval'] = bvals[idx]
        val[idx]['mean'] = np.mean(data[..., idx:idx+1][mask > 0])
        if noise_map:
            val[idx]['std'] = np.std(data_noisemap[mask > 0])
        else:
            val[idx]['std'] = np.std(data[..., idx:idx+1][noise_mask > 0])
            if val[idx]['std'] == 0:
                raise ValueError('Your noise mask does not capture any data'
                                 '(std=0). Please check your noise mask.')

        val[idx]['snr'] = val[idx]['mean'] / val[idx]['std']

    return val
