# -*- coding: utf-8 -*-

"""
Utility operations provided for scil_image_math.py
and scil_connectivity_math.py
They basically act as wrappers around numpy to avoid installing MRtrix/FSL
to apply simple operations on nibabel images or numpy arrays.
"""


from itertools import combinations
from collections import OrderedDict

import nibabel as nib
import numpy as np
from numpy.lib import stride_tricks
from scipy.ndimage import (binary_closing, binary_dilation,
                           binary_erosion, binary_opening,
                           gaussian_filter)
from skimage.filters import threshold_otsu

from scilpy.utils.util import is_float


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
        ('correlation', correlation),
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


def _validate_imgs(*imgs):
    """Make sure that all inputs are images, and that their shapes match."""
    ref_img = imgs[-1]
    for img in imgs:
        if isinstance(img, nib.Nifti1Image) and \
                not np.all(ref_img.header.get_data_shape() ==
                           img.header.get_data_shape()):
            raise ValueError('Not all inputs have the same shape!')


def _validate_imgs_concat(*imgs):
    """Make sure that all inputs are images."""
    for img in imgs:
        if not isinstance(img, nib.Nifti1Image):
            raise ValueError('Inputs are not all images')


def _validate_length(input_list, length, at_least=False):
    """Make sure the the input list has the right number of arguments
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
    """Make sure that the input can be casted to a float."""
    if not is_float(x):
        raise ValueError('The input must be float/int for this operation.')


def cut_up_cube(data, blck):
    """
    cut_up_cube: DATA BLOCK STRIDE
        Cut up a cube of data into patches.
        - blck is the size of the patches.
        - strd is the stride between patches.
        The last cube will be padded with zeros to ensure identical dimensions.
    """
    strd = 1
    pad_size = (blck[0] - 1) // 2
    data = np.pad(data, (pad_size, pad_size),
                  'constant', constant_values=(0, 0))
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data = stride_tricks.as_strided(data, strides=strides, shape=dims)

    return data


def lower_threshold_otsu(input_list, ref_img):
    """
    lower_threshold_otsu: IMG
        All values below or equal to the Otsu threshold will be set to zero.
        All values above the Otsu threshold will be set to one.
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
    _validate_imgs(*input_list, ref_img)

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
    _validate_imgs(*input_list, ref_img)

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
    _validate_imgs(*input_list, ref_img)

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
    _validate_imgs(*input_list, ref_img)

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
    _validate_imgs(*input_list, ref_img)

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
    _validate_imgs(*input_list, ref_img)

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
    output_data = addition(input_list, ref_img)
    output_data[output_data != 0] = 1

    return output_data


def intersection(input_list, ref_img):
    """
    intersection: IMGs
        Operation on binary image to keep the voxels, that are non-zero,
        are present in all files.
    """
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
    _validate_imgs(*input_list, ref_img)

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

    _validate_imgs_concat(*input_list, ref_img)
    if len(input_list[0].header.get_data_shape()) > 4:
        raise ValueError('Concatenate require 3D or 4D arrays.')

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


def correlation(input_list, ref_img, patch_radius=1):
    """
    correlation: IMGs
        Compute the correlation average of multiple images.
    """
    _validate_length(input_list, 2, at_least=True)

    if isinstance(input_list[0], nib.Nifti1Image):
        _validate_imgs(*input_list, ref_img)
        data_shape = input_list[0].header.get_data_shape()
    else:
        data_shape = input_list[0].shape

    sizes = (patch_radius * 2 + 1, patch_radius * 2 + 1, patch_radius * 2 + 1)
    combs = list(combinations(range(len(input_list)), r=2))
    all_corr = np.zeros(data_shape + (len(combs),), dtype=np.float32)

    np.random.seed(0)
    for i, comb in enumerate(combs):
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

        patches_1 = cut_up_cube(data_1, sizes)
        patches_2 = cut_up_cube(data_2, sizes)

        patches_shape = patches_1.shape
        nb_patches = np.prod(patches_shape[0:3])

        patches_1 = patches_1.reshape((nb_patches, np.prod(sizes)))
        patches_2 = patches_2.reshape((nb_patches, np.prod(sizes)))
        patches = np.concatenate((patches_1, patches_2), axis=-1)

        non_zeros_patches = np.sum(patches, axis=-1)
        non_zeros_ids = np.where(np.abs(non_zeros_patches) > 0.001)[0]

        def correlate(data):
            data += np.random.rand(data.shape[0]) * 0.001
            a, b = np.split(data, 2)

            if np.allclose(a, b):
                return 1

            corr = np.corrcoef(a, b, dtype=np.float32)[0, 1]
            return corr

        results = np.zeros((len(patches)), dtype=np.float32)
        results[non_zeros_ids] = np.apply_along_axis(correlate, 1,
                                                     patches[non_zeros_ids, :])

        all_corr[..., i] = results.reshape(patches_shape[0:3])

    return np.mean(all_corr, axis=-1)


def dilation(input_list, ref_img):
    """
    dilation: IMG, VALUE
        Binary morphological operation to spatially extend the values of an
        image to their neighbors. VALUE is in voxels.
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
        Apply a gaussian blur to a single image.
    """
    _validate_length(input_list, 2)
    _validate_type(input_list[0], nib.Nifti1Image)
    _validate_float(input_list[1])

    # Data is always 3D, using directly scipy. See also
    # scilpy.image.volume_operations : smooth_to_fwhm.
    return gaussian_filter(input_list[0].get_fdata(dtype=np.float64),
                           sigma=input_list[1])
