# -*- coding: utf-8 -*-

"""
Utility operations provided for scil_image_math.py
and scil_connectivity_math.py

They basically act as wrappers around numpy to avoid installing MRtrix/FSL
to apply simple operations on nibabel images or numpy arrays.

Docstrings here are NOT typical docstrings: they are the doc printed with the
--help argument in those scripts.

Headers compatibility is NOT verified here. Verified in the main scripts.
"""
import logging
from itertools import combinations
from collections import OrderedDict

import nibabel as nib
import numpy as np
from dipy.io.utils import is_header_compatible
from numpy.lib import stride_tricks
from scipy.ndimage import (binary_closing, binary_dilation,
                           binary_erosion, binary_opening,
                           gaussian_filter)
from skimage.filters import threshold_otsu

from scilpy.utils import is_float


EPSILON = np.finfo(float).eps


def get_array_ops():
    """Get a dictionary of all functions relating to array operations"""
    return OrderedDict([
        ('lower_threshold', lower_threshold),
        ('upper_threshold', upper_threshold),
        ('lower_threshold_eq', lower_threshold_eq),
        ('upper_threshold_eq', upper_threshold_eq),
        ('lower_threshold_otsu', lower_threshold_otsu),
        ('upper_threshold_otsu', upper_threshold_otsu),
        ('lower_clip', lower_clip),
        ('upper_clip', upper_clip),
        ('absolute_value', absolute_value),
        ('round', around),
        ('ceil', ceil),
        ('floor', floor),
        ('normalize_sum', normalize_sum),
        ('normalize_max', normalize_max),
        ('log_10', base_10_log),
        ('log_e', natural_log),
        ('convert', convert),
        ('invert', invert),
        ('addition', addition),
        ('subtraction', subtraction),
        ('multiplication', multiplication),
        ('division', division),
        ('mean', mean),
        ('std', std),
        ('correlation', neighborhood_correlation),
        ('union', union),
        ('intersection', intersection),
        ('difference', difference),
    ])


def get_image_ops():
    """Get a dictionary of all functions relating to image operations"""
    image_ops = get_array_ops()
    image_ops.update(OrderedDict([
        ('concatenate', concatenate),
        ('dilation', dilation),
        ('erosion', erosion),
        ('closing', closing),
        ('opening', opening),
        ('blur', gaussian_blur)
    ]))
    return image_ops


def get_operations_doc(ops: dict):
    """From a dictionary mapping operation names to functions, fetch and join
    all documentations, using the provided names."""
    full_doc = []
    for func in ops.values():
        full_doc.append(func.__doc__)
    return "".join(full_doc)


def _validate_same_shape(*imgs):
    """Make sure that all shapes match."""
    ref_img = imgs[-1]
    for img in imgs:
        if not np.all(ref_img.header.get_data_shape() ==
                      img.header.get_data_shape()):
            raise ValueError('Not all inputs have the same shape!')


def _validate_imgs_type(*imgs):
    """Make sure that all inputs are images."""
    for img in imgs:
        if not isinstance(img, nib.Nifti1Image):
            raise ValueError('Inputs are not all images. Received a {} when '
                             'we were expecting an image.'.format(type(img)))


def _validate_length(input_list, length, at_least=False):
    """Make sure that the input list has the right number of arguments
    (length)."""
    if at_least:
        if not len(input_list) >= length:
            raise ValueError('This operation requires at least {}'
                             ' operands.'.format(length))

    else:
        if not len(input_list) == length:
            raise ValueError('This operation requires exactly {} '
                             'operands.'.format(length))


def _validate_type(x, dtype):
    """Make sure that the input has the right type."""
    if not isinstance(x, dtype):
        raise ValueError('The input must be of type {} for this'
                         ' operation.'.format(dtype))


def _validate_float(x):
    """Make sure that the input can be cast to a float."""
    if not is_float(x):
        raise ValueError('The input must be float/int for this operation.')


def _get_neighbors(data, radius):
    """
    For each voxel of the input data, returns a patch of the local
    neighborhood. Data is padded with zeros for the neighborhood of border
    voxels.

    Parameters
    ----------
    data: np.ndarray
        The data of shape [X, Y, Z].
    radius: int
        Neighborhoods will be cubes of size M = 2 * patch_radius + 1 centered
        at each voxel.

    Returns
    -------
    data: np.ndarray
        The shape of the output will be 6D: [X, Y, Z, M, M, M]
        Ex: data[0, 0, 0, :, :, :] is the neighborhood at voxel [0, 0, 0].
    """
    patch_size = 2 * radius + 1

    # Padding first dimension to have 2 patches fitting on first dimension.
    # (If pad_size is 0, will at least ensure that data is a np.array.)
    pad_size = (patch_size - 1) // 2
    data = np.pad(data, (pad_size, pad_size),
                  mode='constant', constant_values=(0, 0))

    # Preparing types for the call.
    padded_shape = np.array(data.shape)
    patch_size = np.asanyarray([patch_size] * 3)

    # Note. np.r_: Stacks arrays along their first axis.

    # Preparing strides: 6 dimensions.  [sx*strd, sy*strd, sz*strd, sx, sy, sz]
    strides = np.r_[data.strides, data.strides]

    # Preparing dims: 6 dimensions. [nb1, nb2, nb3, B1, B2, B3]
    nbl = (padded_shape - patch_size) + 1
    dims = np.r_[nbl, patch_size]

    # Splitting data
    data = stride_tricks.as_strided(data, strides=strides, shape=dims)

    return data


def lower_threshold_otsu(input_list, ref_img):
    """
    lower_threshold_otsu: IMG
        All values below or equal to the Otsu threshold will be set to zero.
        All values above the Otsu threshold will be set to one.
        (Otsu's method is an algorithm to perform automatic image thresholding
         of the background.)
    """
    _validate_length(input_list, 1)
    _validate_type(input_list[0], nib.Nifti1Image)

    output_data = np.zeros(ref_img.header.get_data_shape(), dtype=np.float64)
    data = input_list[0].get_fdata(dtype=np.float64)
    threshold = threshold_otsu(data)

    output_data[data <= threshold] = 0
    output_data[data > threshold] = 1

    return output_data


def upper_threshold_otsu(input_list, ref_img):
    """
    upper_threshold_otsu: IMG
        All values below the Otsu threshold will be set to one.
        All values above or equal to the Otsu threshold will be set to zero.
        Equivalent to lower_threshold_otsu followed by an inversion.
    """
    _validate_length(input_list, 1)
    _validate_type(input_list[0], nib.Nifti1Image)

    output_data = np.zeros(ref_img.header.get_data_shape(), dtype=np.float64)
    data = input_list[0].get_fdata(dtype=np.float64)
    threshold = threshold_otsu(data)

    output_data[data < threshold] = 1
    output_data[data >= threshold] = 0

    return output_data


def lower_threshold_eq(input_list, ref_img):
    """
    lower_threshold_eq: IMG THRESHOLD
        All values below the threshold will be set to zero.
        All values above or equal the threshold will be set to one.
    """
    _validate_length(input_list, 2)
    _validate_type(input_list[0], nib.Nifti1Image)
    _validate_float(input_list[1])

    output_data = np.zeros(ref_img.header.get_data_shape(), dtype=np.float64)
    data = input_list[0].get_fdata(dtype=np.float64)
    output_data[data < input_list[1]] = 0
    output_data[data >= input_list[1]] = 1

    return output_data


def upper_threshold_eq(input_list, ref_img):
    """
    upper_threshold_eq: IMG THRESHOLD
        All values below or equal the threshold will be set to one.
        All values above the threshold will be set to zero.
        Equivalent to lower_threshold followed by an inversion.
    """
    _validate_length(input_list, 2)
    _validate_type(input_list[0], nib.Nifti1Image)
    _validate_float(input_list[1])

    output_data = np.zeros(ref_img.header.get_data_shape(), dtype=np.float64)
    data = input_list[0].get_fdata(dtype=np.float64)
    output_data[data <= input_list[1]] = 1
    output_data[data > input_list[1]] = 0

    return output_data


def lower_threshold(input_list, ref_img):
    """
    lower_threshold: IMG THRESHOLD
        All values below the threshold will be set to zero.
        All values above the threshold will be set to one.
    """
    _validate_length(input_list, 2)
    _validate_type(input_list[0], nib.Nifti1Image)
    _validate_float(input_list[1])

    output_data = np.zeros(ref_img.header.get_data_shape(), dtype=np.float64)
    data = input_list[0].get_fdata(dtype=np.float64)
    output_data[data <= input_list[1]] = 0
    output_data[data > input_list[1]] = 1

    return output_data


def upper_threshold(input_list, ref_img):
    """
    upper_threshold: IMG THRESHOLD
        All values below the threshold will be set to one.
        All values above the threshold will be set to zero.
        Equivalent to lower_threshold followed by an inversion.
    """
    _validate_length(input_list, 2)
    _validate_type(input_list[0], nib.Nifti1Image)
    _validate_float(input_list[1])

    output_data = np.zeros(ref_img.header.get_data_shape(), dtype=np.float64)
    data = input_list[0].get_fdata(dtype=np.float64)
    output_data[data < input_list[1]] = 1
    output_data[data >= input_list[1]] = 0

    return output_data


def lower_clip(input_list, ref_img):
    """
    lower_clip: IMG THRESHOLD
        All values below the threshold will be set to threshold.
    """
    _validate_length(input_list, 2)
    _validate_type(input_list[0], nib.Nifti1Image)
    _validate_float(input_list[1])

    return np.clip(input_list[0].get_fdata(dtype=np.float64),
                   input_list[1], None)


def upper_clip(input_list, ref_img):
    """
    upper_clip: IMG THRESHOLD
        All values above the threshold will be set to threshold.
    """
    _validate_length(input_list, 2)
    _validate_type(input_list[0], nib.Nifti1Image)
    _validate_float(input_list[1])

    return np.clip(input_list[0].get_fdata(dtype=np.float64),
                   None, input_list[1])


def absolute_value(input_list, ref_img):
    """
    absolute_value: IMG
        All negative values will become positive.
    """
    _validate_length(input_list, 1)
    _validate_type(input_list[0], nib.Nifti1Image)

    return np.abs(input_list[0].get_fdata(dtype=np.float64))


def around(input_list, ref_img):
    """
    round: IMG
        Round all decimal values to the closest integer.
    """
    _validate_length(input_list, 1)
    _validate_type(input_list[0], nib.Nifti1Image)

    return np.round(input_list[0].get_fdata(dtype=np.float64))


def ceil(input_list, ref_img):
    """
    ceil: IMG
        Ceil all decimal values to the next integer.
    """
    _validate_length(input_list, 1)
    _validate_type(input_list[0], nib.Nifti1Image)

    return np.ceil(input_list[0].get_fdata(dtype=np.float64))


def floor(input_list, ref_img):
    """
    floor: IMG
        Floor all decimal values to the previous integer.
    """
    _validate_length(input_list, 1)
    _validate_type(input_list[0], nib.Nifti1Image)

    return np.floor(input_list[0].get_fdata(dtype=np.float64))


def normalize_sum(input_list, ref_img):
    """
    normalize_sum: IMG
        Normalize the image so the sum of all values is one.
    """
    _validate_length(input_list, 1)
    _validate_type(input_list[0], nib.Nifti1Image)

    data = input_list[0].get_fdata(dtype=np.float64)
    return data / np.sum(data)


def normalize_max(input_list, ref_img):
    """
    normalize_max: IMG
        Normalize the image so the maximum value is one.
    """
    _validate_length(input_list, 1)
    _validate_type(input_list[0], nib.Nifti1Image)

    data = input_list[0].get_fdata(dtype=np.float64)
    return data / np.max(data)


def base_10_log(input_list, ref_img):
    """
    log_10: IMG
        Apply a log (base 10) to all non zeros values of an image.
    """
    _validate_length(input_list, 1)
    _validate_type(input_list[0], nib.Nifti1Image)

    data = input_list[0].get_fdata(dtype=np.float64)
    output_data = np.zeros(data.shape, dtype=np.float64)
    output_data[data > EPSILON] = np.log10(data[data > EPSILON])
    output_data[np.abs(output_data) < EPSILON] = -65536

    return output_data


def natural_log(input_list, ref_img):
    """
    log_e: IMG
        Apply a natural log to all non zeros values of an image.
    """
    _validate_length(input_list, 1)
    _validate_type(input_list[0], nib.Nifti1Image)

    data = input_list[0].get_fdata(dtype=np.float64)
    output_data = np.zeros(data.shape, dtype=np.float64)
    output_data[data > EPSILON] = np.log(data[data > EPSILON])
    output_data[np.abs(output_data) < EPSILON] = -65536

    return output_data


def convert(input_list, ref_img):
    """
    convert: IMG
        Perform no operation, but simply change the data type.
    """
    _validate_length(input_list, 1)
    _validate_type(input_list[0], nib.Nifti1Image)

    return input_list[0].get_fdata(dtype=np.float64)


def addition(input_list, ref_img):
    """
    addition: IMGs
        Add multiple images together.
    """
    _validate_length(input_list, 2, at_least=True)
    _validate_imgs_type(*input_list)
    _validate_same_shape(*input_list, ref_img)

    output_data = np.zeros(ref_img.header.get_data_shape(), dtype=np.float64)
    for img in input_list:
        if isinstance(img, nib.Nifti1Image):
            data = img.get_fdata(dtype=np.float64)
            output_data += data
            img.uncache()
        else:
            output_data += img

    return output_data


def subtraction(input_list, ref_img):
    """
    subtraction: IMG_1 IMG_2
        Subtract first image by the second (IMG_1 - IMG_2).
    """
    _validate_length(input_list, 2)
    _validate_imgs_type(*input_list)
    _validate_same_shape(*input_list, ref_img)

    output_data = np.zeros(ref_img.header.get_data_shape(), dtype=np.float64)
    if isinstance(input_list[0], nib.Nifti1Image):
        data_1 = input_list[0].get_fdata(dtype=np.float64)
    else:
        data_1 = input_list[0]
    if isinstance(input_list[1], nib.Nifti1Image):
        data_2 = input_list[1].get_fdata(dtype=np.float64)
    else:
        data_2 = input_list[1]

    output_data += data_1
    return output_data - data_2


def multiplication(input_list, ref_img):
    """
    multiplication: IMGs
        Multiply multiple images together (danger of underflow and overflow)
    """
    _validate_length(input_list, 2, at_least=True)
    _validate_imgs_type(*input_list)
    _validate_same_shape(*input_list, ref_img)

    output_data = np.ones(ref_img.header.get_data_shape())
    if isinstance(input_list[0], nib.Nifti1Image):
        output_data *= input_list[0].get_fdata(dtype=np.float64)
    else:
        output_data *= input_list[0]
    for img in input_list[1:]:
        if isinstance(img, nib.Nifti1Image):
            data = img.get_fdata(dtype=np.float64)
            output_data *= data
            img.uncache()
        else:
            output_data *= img

    return output_data


def division(input_list, ref_img):
    """
    division: IMG_1 IMG_2
        Divide first image by the second (danger of underflow and overflow)
        Ignore zeros values, excluded from the operation.
    """
    _validate_length(input_list, 2)
    _validate_imgs_type(*input_list)
    _validate_same_shape(*input_list, ref_img)

    output_data = np.zeros(ref_img.header.get_data_shape(), dtype=np.float64)
    if isinstance(input_list[0], nib.Nifti1Image):
        data_1 = input_list[0].get_fdata(dtype=np.float64)
    else:
        data_1 = input_list[0]
    if isinstance(input_list[1], nib.Nifti1Image):
        data_2 = input_list[1].get_fdata(dtype=np.float64)
    else:
        data_2 = input_list[1]

    output_data += data_1
    output_data[data_2 != 0] /= data_2[data_2 != 0]
    return output_data


def mean(input_list, ref_img):
    """
    mean: IMGs
        Compute the mean of images.
        If a single 4D image is provided, average along the last dimension.
    """
    _validate_length(input_list, 1, at_least=True)
    _validate_imgs_type(*input_list)
    _validate_same_shape(*input_list, ref_img)

    if len(input_list[0].header.get_data_shape()) > 3:
        if not len(input_list) == 1:
            raise ValueError(
                'This operation with 4D data only support one operand.')
    else:
        if len(input_list) == 1:
            raise ValueError(
                'This operation with only one operand requires 4D data.')

    if len(input_list[0].header.get_data_shape()) > 3:
        return np.average(input_list[0].get_fdata(dtype=np.float64), axis=-1)
    else:
        return addition(input_list, ref_img) / len(input_list)


def std(input_list, ref_img):
    """
    std: IMGs
        Compute the standard deviation average of multiple images.
        If a single 4D image is provided, compute the STD along the last
        dimension.
    """
    _validate_length(input_list, 1, at_least=True)
    _validate_imgs_type(*input_list)
    _validate_same_shape(*input_list, ref_img)

    if len(input_list[0].header.get_data_shape()) > 3:
        if not len(input_list) == 1:
            raise ValueError(
                'This operation with 4D data only support one operand.')
    else:
        if len(input_list) == 1:
            raise ValueError(
                'This operation with only one operand requires 4D data.')

    if len(input_list[0].header.get_data_shape()) > 3:
        return np.std(input_list[0].get_fdata(dtype=np.float64), axis=-1)
    else:
        mean_data = mean(input_list, ref_img)
        output_data = np.zeros(input_list[0].header.get_data_shape())
        for img in input_list:
            if isinstance(img, nib.Nifti1Image):
                data = img.get_fdata(dtype=np.float64)
                output_data += (data - mean_data) ** 2
                img.uncache()
            else:
                output_data += (img - mean_data) ** 2
        return np.sqrt(output_data / len(input_list))


def union(input_list, ref_img):
    """
    union: IMGs
        Operation on binary image to keep voxels, that are non-zero, in at
        least one file.
    """
    # Tests and checks done in addition.
    output_data = addition(input_list, ref_img)
    output_data[output_data != 0] = 1

    return output_data


def intersection(input_list, ref_img):
    """
    intersection: IMGs
        Operation on binary image to keep the voxels, that are non-zero,
        are present in all files.
    """
    # Tests and checks done in multiplication
    output_data = multiplication(input_list, ref_img)
    output_data[output_data != 0] = 1

    return output_data


def difference(input_list, ref_img):
    """
    difference: IMG_1 IMG_2
        Operation on binary image to keep voxels from the first file that are
        not in the second file (non-zeros).
    """
    _validate_length(input_list, 2)
    _validate_imgs_type(*input_list)
    _validate_same_shape(*input_list, ref_img)

    output_data = np.zeros(ref_img.header.get_data_shape(), dtype=np.float64)
    if isinstance(input_list[0], nib.Nifti1Image):
        data_1 = input_list[0].get_fdata(dtype=np.float64)
    else:
        data_1 = input_list[0]
    if isinstance(input_list[1], nib.Nifti1Image):
        data_2 = input_list[1].get_fdata(dtype=np.float64)
    else:
        data_2 = input_list[1]

    output_data[data_1 != 0] = 1
    output_data[data_2 != 0] = 0
    return output_data


def invert(input_list, ref_img):
    """
    invert: IMG
        Operation on binary image to interchange 0s and 1s in a binary mask.
    """
    _validate_length(input_list, 1)
    _validate_type(input_list[0], nib.Nifti1Image)

    data = input_list[0].get_fdata(dtype=np.float64)
    output_data = np.zeros(data.shape, dtype=np.float64)
    output_data[data != 0] = 0
    output_data[data == 0] = 1

    return output_data


def concatenate(input_list, ref_img):
    """
    concatenate: IMGs
        Concatenate a list of 3D and 4D images into a single 4D image.
    """
    _validate_imgs_type(*input_list, ref_img)
    if len(input_list[0].header.get_data_shape()) > 4:
        raise ValueError('Concatenate requires 3D or 4D arrays, but got '
                         '{}'.format(input_list[0].header.get_data_shape()))

    input_data = []
    for img in input_list:

        data = img.get_fdata(dtype=np.float64)

        if len(img.header.get_data_shape()) == 4:
            data = np.rollaxis(data, 3)
            for i in range(0, len(data)):
                input_data.append(data[i])
        else:
            input_data.append(data)

        img.uncache()

    return np.rollaxis(np.stack(input_data), axis=0, start=4)


def _corrcoef_no_nan(data):
    """
    Correlation between two small data arrays, but with our management of NaNs.
    See explanation in neighborhood_correlation docstring.

    Parameters
    ----------
    data: np.array of shape (patch_size**3 * 2, )
        The concatenated, flattened, vectors:
        [data1_flattened, data2_flattened]
    """
    a, b = np.split(data, 2)
    eps = 1e-6

    # If, in at least one patch, all values are the same, we get NaN.
    # Ex: compare a patch of ones with a patch of twos:
    # >> np.corrcoef(np.ones(27), 2*np.ones(27))
    # We chose to return:
    # - 0 if at least one neighborhood was entirely containing background
    #  (note that here, we will never have both backgrounds because of the
    #  indices in the function call, but it is still covered)
    # - 1 if the voxel's neighborhoods are uniform in both images (ex, uniform
    #  gray matter in both images).
    # - 0 if the voxel's neighborhoods is uniform in one image, but not the
    # other (ex, uniform gray matter in a, noisy gray matter in b).
    is_background_a = not np.any(a)
    is_background_b = not np.any(b)
    if is_background_a or is_background_b:
        return 0.0
    if np.std(a) < eps and np.std(b) < eps:
        # Both uniform and non-background
        return 1.0
    elif np.std(a) < eps or np.std(b) < eps:
        # Only one is uniform
        return 0.0

    # If we reach here, corrcoef should not return a NaN
    corr = np.corrcoef(a, b, dtype=np.float32)[0, 1]
    return corr


def neighborhood_correlation(input_list, ref_img):
    """
    correlation: IMGs
        Computes the correlation of the 3x3x3 neighborhood of each voxel, for
        all pair of input images. The final image is the average correlation
        (through all pairs).
        For a given pair of images
        - Background is considered as 0. May lead to very high correlations
        close to the border of the background regions, or very poor ones if the
        background in both images differ.
        - Images are zero-padded. For the same reason as higher, may lead to
        very high correlations if you have data close to the border of the
        image.
        - NaN values (if a voxel's neighborhood is entirely uniform; std 0) are
        replaced by
           - 0 if at least one neighborhood was entirely containing background.
           - 1 if the voxel's neighborhoods are uniform in both images
           - 0 if the voxel's neighborhoods is uniform in one image, but not
           the other.

        UPDATE AS OF VERSION 2.0: Random noise was previously added in the
        process to help avoid NaN values. Now replaced by either 0 or 1 as
        explained above.
    """
    _validate_length(input_list, 2, at_least=True)
    _validate_imgs_type(*input_list, ref_img)
    _validate_same_shape(*input_list, ref_img)
    return neighborhood_correlation_(input_list)


def neighborhood_correlation_(input_list):
    """
    Same as above (neighborhood_correlation) but without the verifications
    required for scil_volume_math.py.

    input_list can be a list of images or a list of arrays.
    """
    data_shape = input_list[0].shape
    combs = list(combinations(range(len(input_list)), r=2))
    all_corr = np.zeros(data_shape + (len(combs),), dtype=np.float32)

    patch_radius = 1  # Using a 3x3x3 neighborhood. Slow enough as it is.
    patch_size = 2 * patch_radius + 1
    np.random.seed(0)

    # For each pair of input images:
    # Possibly loads images twice. Other option is to load all images in
    # memory at once.
    for i, comb in enumerate(combs):
        logging.debug("Computing correlation map for one pair of input "
                      "images.")
        img_1 = input_list[comb[0]]
        img_2 = input_list[comb[1]]

        if isinstance(img_1, nib.Nifti1Image):
            data_1 = img_1.get_fdata(dtype=np.float32)
        else:
            data_1 = img_1
        if isinstance(img_2, nib.Nifti1Image):
            data_2 = img_2.get_fdata(dtype=np.float32)
        else:
            data_2 = img_2

        patches_1 = _get_neighbors(data_1, patch_radius)
        patches_2 = _get_neighbors(data_2, patch_radius)

        patches_shape = patches_1.shape
        nb_patches = np.prod(patches_shape[0:3])

        patches_1 = patches_1.reshape((nb_patches, patch_size ** 3))
        patches_2 = patches_2.reshape((nb_patches, patch_size ** 3))
        patches = np.concatenate((patches_1, patches_2), axis=-1)

        # Removing union of background from data. Already managed in the
        # _corrcoef_no_nan method, but accelerating the process
        non_zeros_patches = np.sum(patches, axis=-1)
        non_zeros_ids = np.where(np.abs(non_zeros_patches) > 1e-6)[0]

        results = np.zeros((len(patches)), dtype=np.float32)

        tmp = np.apply_along_axis(
            _corrcoef_no_nan, axis=1, arr=patches[non_zeros_ids, :])
        results[non_zeros_ids] = tmp

        all_corr[..., i] = results.reshape(patches_shape[0:3])

    return np.mean(all_corr, axis=-1)


def dilation(input_list, ref_img):
    """
    dilation: IMG, VALUE
        Binary morphological operation to spatially extend the values of an
        image to their neighbors. VALUE is in voxels.
        If VALUE is 0, the dilation is repeated until the result does not
        change anymore.
    """
    _validate_length(input_list, 2)
    _validate_type(input_list[0], nib.Nifti1Image)
    _validate_float(input_list[1])

    return binary_dilation(input_list[0].get_fdata(dtype=np.float64),
                           iterations=int(input_list[1]))


def erosion(input_list, ref_img):
    """
    erosion: IMG, VALUE
        Binary morphological operation to spatially shrink the volume contained
        in a binary image. VALUE is in voxels.
        If VALUE is 0, the erosion is repeated until the result does not
        change anymore.
    """
    _validate_length(input_list, 2)
    _validate_type(input_list[0], nib.Nifti1Image)
    _validate_float(input_list[1])

    return binary_erosion(input_list[0].get_fdata(dtype=np.float64),
                          iterations=int(input_list[1]))


def closing(input_list, ref_img):
    """
    closing: IMG, VALUE
        Binary morphological operation, dilation followed by an erosion.
    """
    _validate_length(input_list, 2)
    _validate_type(input_list[0], nib.Nifti1Image)
    _validate_float(input_list[1])

    return binary_closing(input_list[0].get_fdata(dtype=np.float64),
                          iterations=int(input_list[1]))


def opening(input_list, ref_img):
    """
    opening: IMG, VALUE
        Binary morphological operation, erosion followed by a dilation.
    """
    _validate_length(input_list, 2)
    _validate_type(input_list[0], nib.Nifti1Image)
    _validate_float(input_list[1])

    return binary_opening(input_list[0].get_fdata(dtype=np.float64),
                          iterations=int(input_list[1]))


def gaussian_blur(input_list, ref_img):
    """
    blur: IMG, VALUE
        Apply a gaussian blur to a single image. VALUE is sigma, the standard
        deviation of the Gaussian kernel.
    """
    _validate_length(input_list, 2)
    _validate_type(input_list[0], nib.Nifti1Image)
    _validate_float(input_list[1])

    # Data is always 3D, using directly scipy. See also
    # scilpy.image.volume_operations : smooth_to_fwhm.
    return gaussian_filter(input_list[0].get_fdata(dtype=np.float64),
                           sigma=input_list[1])
