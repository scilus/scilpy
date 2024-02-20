# -*- coding: utf-8 -*-

import logging

from dipy.align.imaffine import (AffineMap,
                                 AffineRegistration,
                                 MutualInformationMetric,
                                 transform_centers_of_mass)
from dipy.align.transforms import (AffineTransform3D,
                                   RigidTransform3D)
from dipy.io.utils import get_reference_info
from dipy.reconst.utils import _mask_from_roi, _roi_in_volume

from dipy.segment.mask import crop, median_otsu
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

from scilpy.image.reslice import reslice  # Don't use Dipy's reslice. Buggy.
from scilpy.io.image import get_data_as_mask
from scilpy.gradients.bvec_bval_tools import identify_shells
from scilpy.utils.util import voxel_to_world, world_to_voxel


def count_non_zero_voxels(image):
    """
    Count number of non-zero voxels

    Parameters:
    -----------
    image: string
        Path to the image
    """
    # Count the number of non-zero voxels.
    if len(image.shape) >= 4:
        axes_to_sum = np.arange(3, len(image.shape))
        nb_voxels = np.count_nonzero(np.sum(np.absolute(image),
                                            axis=tuple(axes_to_sum)))
    else:
        nb_voxels = np.count_nonzero(image)

    return nb_voxels


def flip_volume(data, axes):
    """
    Flip volume along a specific axis.

    Parameters
    ----------
    data: np.ndarray
        Volume data.
    axes: List
        A list containing any number of values amongst ['x', 'y', 'z'].

    Return
    ------
    data: np.ndarray
        Flipped volume data along specified axes.
    """
    if 'x' in axes:
        data = data[::-1, ...]

    if 'y' in axes:
        data = data[:, ::-1, ...]

    if 'z' in axes:
        data = data[:, :, ::-1, ...]

    return data


def crop_volume(img: nib.Nifti1Image, wbbox):
    """
    Applies cropping from a world space defined bounding box and fixes the
    affine to keep data aligned.

    Parameters
    ----------
    img: nib.Nifti1Image
        Input image to crop.
    wbbox: WorldBoundingBox
        Bounding box.

    Return
    ------
    nib.Nifti1Image with the cropped data and transformed affine.
    """
    data = img.get_fdata(dtype=np.float32, caching='unchanged')
    affine = img.affine

    voxel_bb_mins = world_to_voxel(wbbox.minimums, affine)
    voxel_bb_maxs = world_to_voxel(wbbox.maximums, affine)

    # Prevent from trying to crop outside data boundaries by clipping bbox
    extent = list(data.shape[:3])
    for i in range(3):
        voxel_bb_mins[i] = max(0, voxel_bb_mins[i])
        voxel_bb_maxs[i] = min(extent[i], voxel_bb_maxs[i])
    translation = voxel_to_world(voxel_bb_mins, affine)

    data_crop = np.copy(crop(data, voxel_bb_mins, voxel_bb_maxs))

    new_affine = np.copy(affine)
    new_affine[0:3, 3] = translation[0:3]

    return nib.Nifti1Image(data_crop, new_affine)


def apply_transform(transfo, reference, moving,
                    interp='linear', keep_dtype=False):
    """
    Apply transformation to an image using Dipy's tool

    Parameters
    ----------
    transfo: numpy.ndarray
        Transformation matrix to be applied
    reference: nib.Nifti1Image
        Filename of the reference image (target)
    moving: nib.Nifti1Image
        Filename of the moving image
    interp : string, either 'linear' or 'nearest'
        the type of interpolation to be used, either 'linear'
        (for k-linear interpolation) or 'nearest' for nearest neighbor
    keep_dtype : bool
        If True, keeps the data_type of the input moving image when saving
        the output image

    Return
    ------
    nib.Nifti1Image of the warped moving image.
    """
    grid2world, dim, _, _ = get_reference_info(reference)
    static_data = reference.get_fdata(dtype=np.float32)

    curr_type = moving.get_data_dtype()
    if keep_dtype:
        moving_data = np.asanyarray(moving.dataobj).astype(curr_type)
    else:
        moving_data = moving.get_fdata(dtype=np.float32)
    moving_affine = moving.affine

    if moving_data.ndim == 3:
        orig_type = moving_data.dtype
        affine_map = AffineMap(np.linalg.inv(transfo),
                               dim, grid2world,
                               moving_data.shape, moving_affine)
        resampled = affine_map.transform(moving_data.astype(np.float64),
                                         interpolation=interp)
    elif moving_data.ndim == 4:
        if isinstance(moving_data[0, 0, 0], np.void):
            raise ValueError('Does not support TrackVis RGB')

        affine_map = AffineMap(np.linalg.inv(transfo),
                               dim[0:3], grid2world,
                               moving_data.shape[0:3], moving_affine)

        orig_type = moving_data.dtype
        resampled = transform_dwi(affine_map, static_data, moving_data,
                                  interpolation=interp)
    else:
        raise ValueError('Does not support this dataset (shape, type, etc)')

    return nib.Nifti1Image(resampled.astype(orig_type), grid2world)


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

    Return
    ------
    nib.Nifti1Image of the warped 4D volume.
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
    dwi: nib.Nifti1Image
        DWI file in nibabel format.
    bval: array
        Array containing bvalues (from dipy.io.gradients.read_bvals_bvecs).
    bvec: array
        Array containing bvectors (from dipy.io.gradients.read_bvals_bvecs).
    b0_thr: int
        Threshold to define b0 minimum value.
    mask: nib.Nifti1Image
        Mask file in nibabel format.
    noise_mask: nib.Nifti1Image
        Noise mask file in nibabel format.
    noise_map: nib.Nifti1Image
        Noise map file in nibabel format.
    basename: string
        Basename used for naming all output files.

    verbose: boolean
        Set to use logging

    Return
    ------
    Dictionary of values (bvec, bval, mean, std, snr) for all volumes.
    """
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    data = dwi.get_fdata(dtype=np.float32)
    affine = dwi.affine
    mask = get_data_as_mask(mask, dtype=bool)

    if split_shells:
        centroids, shell_indices = identify_shells(bval, tol=40.0,
                                                   round_centroids=False,
                                                   sort=False)
        bval = centroids[shell_indices]

    b0s_location = bval <= b0_thr

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
        noise_mask = get_data_as_mask(noise_mask,
                                      dtype=bool).squeeze()
    elif noise_map:
        data_noisemap = noise_map.get_fdata(dtype=np.float32)

    # Val = np array (mean_signal, std_noise)
    val = {0: {'bvec': [0, 0, 0], 'bval': 0, 'mean': 0, 'std': 0}}
    for idx in range(data.shape[-1]):
        val[idx] = {}
        val[idx]['bvec'] = bvec[idx]
        val[idx]['bval'] = bval[idx]
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


def crop_data_with_default_cube(data):
    """ Crop data with a default cube
    Cube: data.shape/3 centered

    Parameters
    ----------
    data : 3D ndarray
        Volume data.
    Returns
    -------
        Data masked
    """
    shape = np.array(data.shape[:3])
    roi_center = shape // 2
    roi_radii = _roi_in_volume(shape, roi_center, shape // 3)
    roi_mask = _mask_from_roi(shape, roi_center, roi_radii)

    return data * roi_mask
