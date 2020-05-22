# -*- coding: utf-8 -*-

"""
Utility operations provided for scil_image_math.py and scil_connectivity_math.py
They basically act as wrappers around numpy to avoid installing MRtrix/FSL
to apply simple operations on nibabel images or numpy arrays.
"""

from collections import OrderedDict
from copy import copy
import logging

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import (binary_closing, binary_dilation,
                                      binary_erosion, binary_opening)

from scilpy.utils.util import is_float


EPSILON = np.finfo(float).eps


def get_array_ops():
    """Get a dictionary of all functions relating to array operations"""
    return OrderedDict([
        ('lower_threshold', lower_threshold),
        ('upper_threshold', upper_threshold),
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
        ('union', union),
        ('intersection', intersection),
        ('difference', difference),
    ])


def get_image_ops():
    """Get a dictionary of all functions relating to image operations"""
    image_ops = get_array_ops()
    image_ops.update(OrderedDict([
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


def _validate_arrays(*arrays):
    """Make sure that all inputs are arrays, and that their shapes match."""
    ref_array = arrays[0]
    for array in arrays:
        if isinstance(array, np.ndarray) and \
                not np.all(ref_array.shape == array.shape):
            raise ValueError('Not all inputs have the same shape!')


def _validate_length(input_list, l, atleast=False):
    """Make sure the the input list has the right number of arguments (l)."""
    if atleast:
        if not len(input_list) >= l:
            logging.error(
                'This operation requires at least {} operands.'.format(l))
            raise ValueError
    else:
        if not len(input_list) == l:
            logging.error(
                'This operation requires exactly {} operands.'.format(l))
            raise ValueError


def _validate_dtype(x, dtype):
    """Make sure that the input has the right datatype."""
    if not isinstance(x, dtype):
        logging.error(
            'The input must be of type {} for this operation.'.format(dtype))
        raise ValueError


def _validate_float(x):
    """Make sure that the input can be casted to a float."""
    if not is_float(x):
        logging.error('The input must be float/int for this operation.')
        raise ValueError


def lower_threshold(input_list):
    """
    lower_threshold: IMG THRESHOLD
        All values below the threshold will be set to zero.
        All values above the threshold will be set to one.
    """
    _validate_length(input_list, 2)
    _validate_dtype(input_list[0], np.ndarray)
    _validate_float(input_list[1])

    output_data = copy(input_list[0])
    output_data[input_list[0] < input_list[1]] = 0
    output_data[input_list[0] >= input_list[1]] = 1

    return output_data


def upper_threshold(input_list):
    """
    upper_threshold: IMG THRESHOLD
        All values below the threshold will be set to one.
        All values above the threshold will be set to zero.
        Equivalent to lower_threshold followed by an inversion.
    """
    _validate_length(input_list, 2)
    _validate_dtype(input_list[0], np.ndarray)
    _validate_float(input_list[1])

    output_data = copy(input_list[0])
    output_data[input_list[0] <= input_list[1]] = 1
    output_data[input_list[0] > input_list[1]] = 0

    return output_data


def lower_clip(input_list):
    """
    lower_clip: IMG THRESHOLD
        All values below the threshold will be set to threshold.
    """
    _validate_length(input_list, 2)
    _validate_dtype(input_list[0], np.ndarray)
    _validate_float(input_list[1])

    return np.clip(input_list[0], input_list[1], None)


def upper_clip(input_list):
    """
    upper_clip: IMG THRESHOLD
        All values above the threshold will be set to threshold.
    """
    _validate_length(input_list, 2)
    _validate_dtype(input_list[0], np.ndarray)
    _validate_float(input_list[1])

    return np.clip(input_list[0], None, input_list[1])


def absolute_value(input_list):
    """
    absolute_value: IMG
        All negative values will become positive.
    """
    _validate_length(input_list, 1)
    _validate_dtype(input_list[0], np.ndarray)

    return np.abs(input_list[0])


def around(input_list):
    """
    round: IMG
        Round all decimal values to the closest integer.
    """
    _validate_length(input_list, 1)
    _validate_dtype(input_list[0], np.ndarray)

    return np.round(input_list[0])


def ceil(input_list):
    """
    ceil: IMG
        Ceil all decimal values to the next integer.
    """
    _validate_length(input_list, 1)
    _validate_dtype(input_list[0], np.ndarray)

    return np.ceil(input_list[0])


def floor(input_list):
    """
    floor: IMG
        Floor all decimal values to the previous integer.
    """
    _validate_length(input_list, 1)
    _validate_dtype(input_list[0], np.ndarray)

    return np.floor(input_list[0])


def normalize_sum(input_list):
    """
    normalize_sum: IMG
        Normalize the image so the sum of all values is one.
    """
    _validate_length(input_list, 1)
    _validate_dtype(input_list[0], np.ndarray)

    return copy(input_list[0]) / np.sum(input_list[0])


def normalize_max(input_list):
    """
    normalize_max: IMG
        Normalize the image so the maximum value is one.
    """
    _validate_length(input_list, 1)
    _validate_dtype(input_list[0], np.ndarray)

    return copy(input_list[0]) / np.max(input_list[0])


def base_10_log(input_list):
    """
    log_10: IMG
        Apply a log (base 10) to all non zeros values of an image.
    """
    _validate_length(input_list, 1)
    _validate_dtype(input_list[0], np.ndarray)

    output_data = np.zeros(input_list[0].shape)
    output_data[input_list[0] > EPSILON] = np.log10(
        input_list[0][input_list[0] > EPSILON])
    output_data[np.abs(output_data) < EPSILON] = -65536

    return output_data


def natural_log(input_list):
    """
    log_e: IMG
        Apply a natural log to all non zeros values of an image.
    """
    _validate_length(input_list, 1)
    _validate_dtype(input_list[0], np.ndarray)

    output_data = np.zeros(input_list[0].shape)
    output_data[input_list[0] > EPSILON] = np.log(
        input_list[0][input_list[0] > EPSILON])
    output_data[np.abs(output_data) < EPSILON] = -65536

    return output_data


def convert(input_list):
    """
    convert: IMG
        Perform no operation, but simply change the data type.
    """
    _validate_length(input_list, 1)
    _validate_dtype(input_list[0], np.ndarray)

    return copy(input_list[0])


def addition(input_list):
    """
    addition: IMGs
        Add multiple images together.
    """
    _validate_length(input_list, 2, atleast=True)
    _validate_arrays(*input_list)
    ref_array = input_list[0]

    output_data = np.zeros(ref_array.shape)
    for data in input_list:
        output_data += data

    return output_data


def subtraction(input_list):
    """
    subtraction: IMG_1 IMG_2
        Subtract first image by the second (IMG_1 - IMG_2).
    """
    _validate_length(input_list, 2)
    _validate_arrays(*input_list)

    return input_list[0] - input_list[1]


def multiplication(input_list):
    """
    multiplication: IMGs
        Multiply multiple images together (danger of underflow and overflow)
    """
    _validate_length(input_list, 2, atleast=True)
    _validate_arrays(*input_list)

    output_data = input_list[0]
    for data in input_list[1:]:
        output_data *= data

    return output_data


def division(input_list):
    """
    division: IMG_1 IMG_2
        Divide first image by the second (danger of underflow and overflow)
        Ignore zeros values, excluded from the operation.
    """
    _validate_length(input_list, 2)
    _validate_arrays(*input_list)
    ref_array = input_list[0]

    output_data = np.zeros(ref_array.shape)
    output_data[input_list[1] != 0] = input_list[0][input_list[1] != 0] \
        / input_list[1][input_list[1] > 0]
    return output_data


def mean(input_list):
    """
    mean: IMGs
        Compute the mean of images.
        If a single 4D image is provided, average along the last dimension.
    """
    _validate_length(input_list, 1, atleast=True)
    _validate_arrays(*input_list)
    ref_array = input_list[0]

    if len(input_list) == 1 and not ref_array.ndim > 3:
        logging.error('This operation with only one operand requires 4D data.')
        raise ValueError

    in_data = np.squeeze(np.rollaxis(np.array(input_list), 0,
                                     input_list[0].ndim+1))

    return np.average(in_data, axis=-1)


def std(input_list):
    """
    std: IMGs
        Compute the standard deviation average of multiple images.
        If a single 4D image is provided, compute the STD along the last
        dimension.
    """
    _validate_length(input_list, 1, atleast=True)
    _validate_arrays(*input_list)
    ref_array = input_list[0]

    if len(input_list) == 1 and not ref_array.ndim > 3:
        logging.error('This operation with only one operand requires 4D data.')
        raise ValueError

    in_data = np.squeeze(np.rollaxis(np.array(input_list), 0,
                                     input_list[0].ndim+1))

    return np.std(in_data, axis=-1)


def union(input_list):
    """
    union: IMGs
        Operation on binary image to keep voxels, that are non-zero, in at
        least one file.
    """
    output_data = addition(input_list)
    output_data[output_data != 0] = 1

    return output_data


def intersection(input_list):
    """
    intersection: IMGs
        Operation on binary image to keep the voxels, that are non-zero,
        are present in all files.
    """
    output_data = multiplication(input_list)
    output_data[output_data != 0] = 1

    return output_data


def difference(input_list):
    """
    difference: IMG_1 IMG_2
        Operation on binary image to keep voxels from the first file that are
        not in the second file (non-zeros).
    """
    _validate_length(input_list, 2)
    _validate_arrays(*input_list)

    output_data = copy(input_list[0]).astype(np.bool)
    output_data[input_list[1] != 0] = 0

    return output_data


def invert(input_list):
    """
    invert: IMG
        Operation on binary image to interchange 0s and 1s in a binary mask.
    """
    _validate_length(input_list, 1)
    _validate_arrays(*input_list)

    output_data = np.zeros(input_list[0].shape)
    output_data[input_list[0] != 0] = 0
    output_data[input_list[0] == 0] = 1

    return output_data


def dilation(input_list):
    """
    dilation: IMG, VALUE
        Binary morphological operation to spatially extend the values of an
        image to their neighbors.
    """
    _validate_length(input_list, 2)
    _validate_arrays(input_list[0])
    _validate_float(input_list[1])

    return binary_dilation(input_list[0], iterations=int(input_list[1]))


def erosion(input_list):
    """
    erosion: IMG, VALUE
        Binary morphological operation to spatially shrink the volume contained
        in a binary image.
    """
    _validate_length(input_list, 2)
    _validate_arrays(input_list[0])
    _validate_float(input_list[1])

    return binary_erosion(input_list[0], iterations=int(input_list[1]))


def closing(input_list):
    """
    closing: IMG, VALUE
        Binary morphological operation, dilation followed by an erosion.
    """
    _validate_length(input_list, 2)
    _validate_arrays(input_list[0])
    _validate_float(input_list[1])

    return binary_closing(input_list[0], iterations=int(input_list[1]))


def opening(input_list):
    """
    opening: IMG, VALUE
        Binary morphological operation, erosion followed by a dilation.
    """
    _validate_length(input_list, 2)
    _validate_arrays(input_list[0])
    _validate_float(input_list[1])

    return binary_opening(input_list[0], iterations=int(input_list[1]))


def gaussian_blur(input_list):
    """
    blur: IMG, VALUE
        Apply a gaussian blur to a single image.
    """
    _validate_length(input_list, 2)
    _validate_arrays(input_list[0])
    _validate_float(input_list[1])

    return gaussian_filter(input_list[0], sigma=input_list[1])
