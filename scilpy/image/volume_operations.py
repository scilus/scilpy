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
from numpy import ma
from scipy.ndimage import binary_dilation, gaussian_filter
from sklearn import linear_model

from scilpy.image.reslice import reslice  # Don't use Dipy's reslice. Buggy.
from scilpy.io.image import get_data_as_mask
from scilpy.gradients.bvec_bval_tools import identify_shells
from scilpy.utils.spatial import voxel_to_world
from scilpy.utils.spatial import world_to_voxel


def count_non_zero_voxels(image):
    """
    Count number of non-zero voxels

    Parameters:
    -----------
    image: np.ndarray
        The loaded image.

    Returns
    -------
    nb_voxels: int
        The count.
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

    Returns
    -------
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

    Returns
    -------
    cropped_im: nib.Nifti1Image
        The image with cropped data and transformed affine.
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

    Returns
    -------
    moved_im: nib.Nifti1Image
        The warped moving image.
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
        else:
            logging.warning('You are applying a transform to a 4D volume. If'
                            'it is a DWI volume, make sure to rotate your '
                            'bvecs with scil_gradients_apply_transform.py')

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

    Returns
    -------
    trans_dwi: nib.Nifti1Image
        The warped 4D volume.
    """
    trans_dwi = np.zeros(static.shape + (dwi.shape[3],), dtype=dwi.dtype)
    for i in range(dwi.shape[3]):
        trans_dwi[..., i] = reg_obj.transform(dwi[..., i],
                                              interpolation=interpolation)

    return trans_dwi


def register_image(static, static_grid2world, moving, moving_grid2world,
                   transformation_type='affine', dwi=None, fine=False):
    """
    Register a moving image to a static image using either rigid or affine
    transformations. If a DWI (4D) is provided, it applies the transformation
    to each volume.

    Parameters
    ----------
    static : ndarray
        The static image volume to which the moving image will be registered.
    static_grid2world : ndarray
        The grid-to-world (vox2ras) transformation associated with the static
        image.
    moving : ndarray
        The moving image volume to register to the static image.
    moving_grid2world : ndarray
        The grid-to-world (vox2ras) transformation associated with the moving
        image.
    transformation_type : str, optional
        The type of transformation ('rigid' or 'affine'). Default is 'affine'.
    dwi : ndarray, optional
        Diffusion-weighted imaging data (if applicable). Default is None.
        If given, then `moving` should be the reference template.
    fine : bool, optional
        Whether to use fine or coarse settings for the registration.
        Default is False.

    Raises
    ------
    ValueError
        If the transformation_type is neither 'rigid' nor 'affine'.

    Returns
    -------
    moved: np.ndarray
        If `dwi` is None, returns the transformed moving image, else the
        transformed dwi.
    transform: np.ndarray
        The transformation matrix.
    """
    if transformation_type not in ['rigid', 'affine']:
        raise ValueError('Transformation type not available in Dipy')

    # Set all parameters for registration
    nbins = 64 if fine else 32
    params0 = None
    sampling_prop = None
    level_iters = [250, 100, 50, 25] if fine else [50, 25, 5]

    # With images too small, dipy fails with no clear warning.
    if (np.any(np.asarray(moving.shape) < 8) or
            np.any(np.asarray(static.shape) < 8)):
        raise ValueError("Current implementation of registration was prepared "
                         "with factors up to 8. Requires images with at least "
                         "8 voxels in each direction.")

    sigmas = [8.0, 4.0, 2.0, 1.0] if fine else [8.0, 4.0, 2.0]
    factors = [8, 4, 2, 1.0] if fine else [8, 4, 2]
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


def compute_snr(dwi, bval, bvec, b0_thr, mask, noise_mask=None, noise_map=None,
                split_shells=False):
    """
    Computes the SNR. One SNR per DWI volume is computed, with
                     SNR = mean(data) / std(noise)
    Where
    - mean is the mean of all DWI voxels inside your given mask.
    - std is the standard deviatation of the noise. For instance, you could
      want to use std of the background. Here, we use:
         - noise_map[mask] if noise_map is provided
         - data[noise_mask] if noise_mask is provided
         - data[automatic_mask] if neither are provided: we will try to
         discover a noise_mask automatically in the background (from the upper
         half, to avoid using neck and shoulder).

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
    noise_mask: nib.Nifti1Image, optional
        Noise mask file in nibabel format. Only one of noise_mask or noise_map
        may be used.
    noise_map: nib.Nifti1Image, optional
        Noise map file in nibabel format. Only one of noise_mask or noise_map
        may be used.
    split_shells: bool
        If true, we will only work with one b-value per shell (the discovered
        centroids).

    Returns
    -------
    val: dict
        Dictionary of values (bvec, bval, mean, std, snr) for all volumes.
    noise_mask: np.ndarray or None
        The noise_mask that was used; either None (if noise_map was given), or
        the given mask, or the discovered mask.
    """
    data = dwi.get_fdata(dtype=np.float32)
    mask = get_data_as_mask(mask, dtype=bool)

    if split_shells:
        centroids, shell_indices = identify_shells(bval, tol=40.0,
                                                   round_centroids=False,
                                                   sort=False)
        bval = centroids[shell_indices]

    b0s_location = bval <= b0_thr

    if not np.any(b0s_location):
        raise ValueError('You should ajust b0_thr (currently {}). No b0 was '
                         'found.'.format(b0_thr))

    if noise_map and noise_mask:
        raise ValueError("Please only use either noise_map or noise_mask, not "
                         "both.")
    elif noise_map:
        data_noisemap = noise_map.get_fdata(dtype=np.float32)
    else:
        if noise_mask is None:
            logging.info("No noise mask given. Trying to discover "
                         "automatically from the upper half of the image "
                         "(typically allowing to exlude neck and shoulder, "
                         "if any).")
            # Note median_otsu is ~BET
            b0_mask, noise_mask = median_otsu(data, vol_idx=b0s_location)

            # we inflate the mask, then invert it to recover only the noise
            noise_mask = binary_dilation(noise_mask, iterations=10).squeeze()

            # Add the upper half in order to delete the neck and shoulder
            # when inverting the mask
            noise_mask[..., :noise_mask.shape[-1]//2] = 1

            # Reverse the mask to get only noise
            noise_mask = (~noise_mask).astype(bool)
            automatic_noise_mask = True
        else:
            noise_mask = get_data_as_mask(noise_mask, dtype=bool).squeeze()
            automatic_noise_mask = False

        logging.info('Number of voxels found in noise mask : {} / {}'
                     .format(np.count_nonzero(noise_mask),
                             np.size(noise_mask)))

    # Val = np array (mean_signal, std_noise)
    val = {0: {'bvec': [0, 0, 0], 'bval': 0, 'mean': 0, 'std': 0}}
    for idx in range(data.shape[-1]):
        val[idx] = {'bvec': bvec[idx],
                    'bval': bval[idx],
                    'mean': np.mean(data[..., idx][mask > 0])}
        if noise_map:
            val[idx]['std'] = np.std(data_noisemap[mask > 0])
        else:
            val[idx]['std'] = np.std(data[..., idx][noise_mask])
            if val[idx]['std'] == 0:
                if automatic_noise_mask:
                    raise ValueError("No noise in the background such as "
                                     "discovered automatically. Please give "
                                     "your own noise_mask for more accuracy.")
                else:
                    raise ValueError('Your noise mask does not capture any '
                                     'noise (std=0). Please check your noise '
                                     'mask.')

        val[idx]['snr'] = val[idx]['mean'] / val[idx]['std']

    return val, noise_mask


def remove_outliers_ransac(in_data, min_fit, fit_thr, max_iter):
    """
    Remove outliers from image using the RANSAC algorithm.

    Parameters
    ----------
    in_data: np.ndarray
        The input.
    min_fit: int
        The minimum number of data values required to fit the model.
    fit_thr: float
        Threshold value for determining when a data point fits a model.
    max_iter: int
        The maximum number of iterations allowed in the algorithm.

    Returns
    -------
    out_data: np.ndarray
        Data without outliers.
    """
    init_shape = in_data.shape
    in_data_flat = in_data.flatten()
    in_nzr_ind = np.nonzero(in_data_flat)
    in_nzr_val = np.array(in_data_flat[in_nzr_ind])

    X = in_nzr_ind[0][:, np.newaxis]
    model_ransac = linear_model.RANSACRegressor(
        base_estimator=linear_model.LinearRegression(), min_samples=min_fit,
        residual_threshold=fit_thr, max_trials=max_iter)
    model_ransac.fit(X, in_nzr_val)

    outlier_mask = np.logical_not(model_ransac.inlier_mask_)
    outliers = X[outlier_mask]
    logging.info('# outliers: {}'.format(len(outliers)))

    in_data_flat[outliers] = 0
    out_data = np.reshape(in_data_flat, init_shape)

    return out_data


def smooth_to_fwhm(data, fwhm):
    """
    Smooth a volume to given FWHM.

    Parameters
    ----------
    data: np.ndarray
        3D or 4D data. If it is 4D, processing invidually on each volume (on
        the last dimension)
    fwhm: float
        Full width at half maximum.
    """
    if fwhm > 0:
        # converting fwhm to Gaussian std
        gauss_std = fwhm / np.sqrt(8 * np.log(2))

        if len(data.shape) == 3:
            data_smooth = gaussian_filter(data, sigma=gauss_std)
        elif len(data.shape) == 4:
            data_smooth = np.zeros(data.shape)
            for v in range(data.shape[-1]):
                data_smooth[..., v] = gaussian_filter(data[..., v],
                                                      sigma=gauss_std)
        else:
            raise ValueError("Expecting a 3D or 4D volume.")

        return data_smooth
    else:
        return data


def _interp_code_to_order(interp_code):
    orders = {'nn': 0, 'lin': 1, 'quad': 2, 'cubic': 3}
    return orders[interp_code]


def resample_volume(img, ref_img=None, volume_shape=None, iso_min=False,
                    voxel_res=None,
                    interp='lin', enforce_dimensions=False):
    """
    Function to resample a dataset to match the resolution of another reference
    dataset or to the resolution specified as in argument.

    One (and only one) of the following options must be chosen:
    ref, volume_shape, iso_min or voxel_res.

    Parameters
    ----------
    img: nib.Nifti1Image
        Image to resample.
    ref_img: nib.Nifti1Image, optional
        Reference volume to resample to. This method is used only if ref is not
        None. (default: None)
    volume_shape: tuple, shape (3,) or int, optional
        Final shape to resample to. If the value it is set to is Y, it will
        resample to an isotropic shape of Y x Y x Y. This method is used
        only if volume_shape is not None. (default: None)
    iso_min: bool, optional
        If true, resample the volume to R x R x R resolution, with R being the
        smallest current voxel dimension. If false, this method is not used.
    voxel_res: tuple, shape (3,) or float, optional
        Set the zoom property of the image at the specified resolution.
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
    original_shape = data.shape
    affine = img.affine
    original_zooms = img.header.get_zooms()[:3]

    error_msg = ('Please only provide one option amongst ref_img, '
                 'volume_shape, voxel_res or iso_min.')
    if ref_img is not None:
        if iso_min or voxel_res or volume_shape:
            raise ValueError(error_msg)
        new_zooms = ref_img.header.get_zooms()[:3]
    elif volume_shape is not None:
        if iso_min or voxel_res:
            raise ValueError(error_msg)
        if len(volume_shape) == 1:
            volume_shape = volume_shape * 3

        new_zooms = tuple((o / v) * z for o, v, z in
                          zip(original_shape, volume_shape, original_zooms))

    elif iso_min:
        if voxel_res:
            raise ValueError(error_msg)
        min_zoom = min(original_zooms)
        new_zooms = (min_zoom, min_zoom, min_zoom)
    elif voxel_res:
        new_zooms = voxel_res
        if len(voxel_res) == 1:
            new_zooms = voxel_res * 3
    else:
        raise ValueError("You must choose the resampling method. Either with"
                         "a reference volume, or a chosen image shape, "
                         "or chosen resolution, or option iso_min.")

    interp_choices = ['nn', 'lin', 'quad', 'cubic']
    if interp not in interp_choices:
        raise ValueError("interp must be one of 'nn', 'lin', 'quad', 'cubic'.")

    logging.info('Data shape: %s', data.shape)
    logging.info('Data affine: %s', affine)
    logging.info('Data affine setup: %s', nib.aff2axcodes(affine))
    logging.info('Resampling data to %s with mode %s', new_zooms, interp)

    data2, affine2 = reslice(data, affine, original_zooms, new_zooms,
                             _interp_code_to_order(interp))

    logging.info('Resampled data shape: %s', data2.shape)
    logging.info('Resampled data affine: %s', affine2)
    logging.info('Resampled data affine setup: %s', nib.aff2axcodes(affine2))

    if enforce_dimensions:
        if ref_img is None:
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


def mask_data_with_default_cube(data):
    """Masks data outside a default cube (Cube: data.shape/3 centered)

    Parameters
    ----------
    data :  np.ndarray
        Volume data, 3D.

    Returns
    -------
    data: np.ndarray
        Data masked
    """
    shape = np.array(data.shape[:3])
    roi_center = shape // 2
    roi_radii = _roi_in_volume(shape, roi_center, shape // 3)
    roi_mask = _mask_from_roi(shape, roi_center, roi_radii)

    return data * roi_mask


def normalize_metric(metric, reverse=False):
    """
    Normalize a metric array to a range between 0 and 1,
    optionally reversing the normalization.

    Parameters
    ----------
    metric : ndarray
        The input metric array to be normalized.
    reverse : bool, optional
        If True, reverse the normalization (i.e., 1 - normalized value).
        Default is False.

    Returns
    -------
    ndarray
        The normalized (and possibly reversed) metric array.
        NaN values in the input are retained.
    """
    mask = np.isnan(metric)
    masked_metric = ma.masked_array(metric, mask)

    min_val, max_val = masked_metric.min(), masked_metric.max()
    normalized_metric = (masked_metric - min_val) / (max_val - min_val)

    if reverse:
        normalized_metric = 1 - normalized_metric

    return ma.filled(normalized_metric, fill_value=np.nan)


def merge_metrics(*arrays, beta=1.0):
    """
    Merge an arbitrary number of metrics into a single heatmap using a weighted
    geometric mean, ignoring NaN values. Each input array contributes equally
    to the geometric mean, and the result is boosted by a specified factor.

    Parameters
    ----------
    *arrays : ndarray
        An arbitrary number of input arrays (ndarrays).
        All arrays must have the same shape.
    beta : float, optional
        Boosting factor for the geometric mean. The default is 1.0.

    Returns
    -------
    ndarray
        Boosted geometric mean of the inputs (same shape as the input arrays)
        NaN values in any input array are propagated to the output.
    """

    # Create a mask for NaN values in any of the arrays
    mask = np.any([np.isnan(arr) for arr in arrays], axis=0)
    masked_arrays = [ma.masked_array(arr, mask) for arr in arrays]

    # Calculate the product of the arrays for the geometric mean
    array_product = np.prod(masked_arrays, axis=0)

    # Calculate the geometric mean for valid data
    geometric_mean = np.power(array_product, 1 / len(arrays))
    boosted_mean = geometric_mean ** beta

    return ma.filled(boosted_mean, fill_value=np.nan)
