# -*- coding: utf-8 -*-

"""
Utility operations provided for scil_image_math.py and scil_connectiity_math.py
They basically act as wrappers around numpy to avoid installing MRtrix/FSL
to apply simple operations on nibabel images or numpy arrays.
"""

from copy import copy
import logging

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import (binary_closing, binary_dilation,
                                      binary_erosion, binary_opening)

from scilpy.utils.util import is_float


def get_array_ops():
    """Get a dictionary of all functions relating to array operations"""
    return {
        'lower_threshold': lower_threshold,
        'upper_threshold': upper_threshold,
        'lower_clip': lower_clip,
        'upper_clip': upper_clip,
        'absolute_value': absolute_value,
        'round': around,
        'ceil': ceil,
        'floor': floor,
        'normalize_sum': normalize_sum,
        'normalize_max': normalize_max,
        'convert': convert,
        'invert': invert,
        'addition': addition,
        'subtraction': subtraction,
        'multiplication': multiplication,
        'division': division,
        'mean': mean,
        'std': std,
        'union': union,
        'intersection': intersection,
        'difference': difference,
    }


def get_image_ops():
    """Get a dictionary of all functions relating to image operations"""
    image_ops = get_array_ops()
    image_ops.update({
        'dilation': dilation,
        'erosion': erosion,
        'closing': closing,
        'opening': opening,
        'blur': gaussian_blur
    })
    return image_ops


def get_operations_doc(ops: dict):
    """From a dictionary mapping operation names to functions, fetch and join
    all documentations, using the provided names."""
    full_doc = []
    for _, func in ops:
        full_doc.append(func.__doc__)
    return "".join(full_doc)


def get_array_operations_doc():
    """Fetch documentation from all array operations."""
    return "".join([f.__doc__ for f in get_array_ops().values()])


def get_image_operations_doc():
    """Fetch documentation from all image operations."""
    return "".join([f.__doc__ for f in get_image_ops().values()])


def find_array(input_list):
    at_least_one_arr = False
    ref_array = None
    for i in input_list:
        if isinstance(i, np.ndarray):
            at_least_one_arr = True
            ref_array = i
            break

    if not at_least_one_arr:
        logging.error('At least one array is required in the input list.')
        raise ValueError

    return ref_array


def lower_threshold(input_list):
    """
    lower_threshold: IMG THRESHOLD
        All values below the threshold will be set to zero.
        All values above the threshold will be set to one.
    """
    if not len(input_list) == 2:
        logging.error('This operation only support two operands.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError
    if not is_float(input_list[1]):
        logging.error('Second input must be float/int for this operation.')
        raise ValueError

    output_data = copy(input_list[0])
    output_data[input_list[0] < input_list[1]] = 0
    output_data[input_list[0] >= input_list[1]] = 1

    return output_data


def upper_threshold(input_list):
    """
    upper_threshold: IMG THRESHOLD
        All values below the threshold will be set to one.
        All values above the threshold will be set to zero.
    """
    if not len(input_list) == 2:
        logging.error('This operation only support two operands.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError
    if not is_float(input_list[1]):
        logging.error('Second input must be float/int for this operation.')
        raise ValueError

    output_data = copy(input_list[0])
    output_data[input_list[0] <= input_list[1]] = 1
    output_data[input_list[0] > input_list[1]] = 0

    return output_data


def lower_clip(input_list):
    """
    lower_clip: IMG THRESHOLD
        All values below the threshold will be set to threshold.
    """
    if not len(input_list) == 2:
        logging.error('This operation only support two operands.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError
    if not is_float(input_list[1]):
        logging.error('Second input must be float/int for this operation.')
        raise ValueError

    return np.clip(input_list[0], input_list[1], None)


def upper_clip(input_list):
    """
    upper_clip: IMG THRESHOLD
        All values above the threshold will be set to threshold.
    """
    if not len(input_list) == 2:
        logging.error('This operation only support two operands.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError
    if not is_float(input_list[1]):
        logging.error('Second input must be float/int for this operation.')
        raise ValueError

    return np.clip(input_list[0], None, input_list[1])


def absolute_value(input_list):
    """
    absolute_value: IMG
        All negative values will become positive.
    """
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    return np.abs(input_list[0])


def around(input_list):
    """
    round: IMG
        Round all decimal values to the closest integer.
    """
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    return np.round(input_list[0])


def ceil(input_list):
    """
    ceil: IMG
        Ceil all decimal values to the next integer.
    """
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    return np.ceil(input_list[0])


def floor(input_list):
    """
    floor: IMG
        Floor all decimal values to the previous integer.
    """
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    return np.floor(input_list[0])


def normalize_sum(input_list):
    """
    normalize_sum: IMG
        Normalize the image so the sum of all values is one.
    """
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    return copy(input_list[0]) / np.sum(input_list[0])


def normalize_max(input_list):
    """
    normalize_max: IMG
        Normalize the image so the maximum value is one.
    """
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    return copy(input_list[0]) / np.max(input_list[0])


def convert(input_list):
    """
    convert: IMG
        Perform no operation, but simply change the data type.
    """
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    return copy(input_list[0])


def addition(input_list):
    """
    addition: IMGs
        Add multiple images together.
    """
    if not len(input_list) > 1:
        logging.error('This operation requires at least two operands.')
        raise ValueError

    ref_array = find_array(input_list)

    output_data = np.zeros(ref_array.shape)
    for data in input_list:
        output_data += data

    return output_data


def subtraction(input_list):
    """
    subtraction: IMG_1 IMG_2
        Subtract two images together (IMG_1 - IMG_2).
    """
    if not len(input_list) == 2:
        logging.error('This operation only support two operands.')
        raise ValueError

    _ = find_array(input_list)

    return input_list[0] - input_list[1]


def multiplication(input_list):
    """
    multiplication: IMGs
        Multiply multiple images together (danger of underflow and overflow)
    """
    if not len(input_list) > 1:
        logging.error('This operation requires at least two operands.')
        raise ValueError

    _ = find_array(input_list)

    output_data = input_list[0]
    for data in input_list[1:]:
        output_data *= data

    return output_data


def division(input_list):
    """
    division: IMG_1 IMG_2
        Divide two images together (danger of underflow and overflow)
    """
    if not len(input_list) == 2:
        logging.error('This operation only support two operands.')
        raise ValueError

    ref_array = find_array(input_list)

    output_data = np.zeros(ref_array.shape)
    output_data[input_list[1] > 0] = input_list[0][input_list[1] > 0] \
        / input_list[1][input_list[1] > 0]
    return output_data


def mean(input_list):
    """
    mean: IMGs
        If a single 4D image is provided, average along the last dimension.
    """
    if not len(input_list) > 0:
        logging.error('This operations required either one operand (4D) or'
                      'or multiple operands (3D/4D).')
        raise ValueError

    ref_array = find_array(input_list)
    for i in input_list:
        if not isinstance(i, np.ndarray):
            logging.error('All inputs must be array.')
            raise ValueError
        if not i.shape == ref_array.shape:
            logging.error('All shapes must match.')
            raise ValueError

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
    if not len(input_list) > 0:
        logging.error('This operations required either one operand (4D) or'
                      'or multiple operands (3D/4D).')
        raise ValueError

    ref_array = find_array(input_list)
    for i in input_list:
        if not isinstance(i, np.ndarray):
            logging.error('All inputs must be array.')
            raise ValueError
        if not i.shape == ref_array.shape:
            logging.error('All shapes must match.')
            raise ValueError

    if len(input_list) == 1 and not ref_array.ndim > 3:
        logging.error('This operation with only one operand requires 4D data.')
        raise ValueError

    in_data = np.squeeze(np.rollaxis(np.array(input_list), 0,
                                     input_list[0].ndim+1))

    return np.std(in_data, axis=-1)


def union(input_list):
    """
    union: IMGs
        Binary operation to keep voxels that are in any file.
    """
    output_data = addition(input_list)
    output_data[output_data != 0] = 1

    return output_data


def intersection(input_list):
    """
    intersection: IMGs
        Binary operation to keep the voxels that are present in all files.
    """
    output_data = multiplication(input_list)
    output_data[output_data != 0] = 1

    return output_data


def difference(input_list):
    """
    difference: IMG_1 IMG_2
        Binary operation to keep voxels from the first file that are not in
        the second file.
    """
    if not len(input_list) == 2:
        logging.error('This operation only support two operands.')
        raise ValueError

    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    if not isinstance(input_list[1], np.ndarray):
        logging.error('.')
        raise ValueError

    output_data = copy(input_list[0]).astype(np.bool)
    output_data[input_list[1] != 0] = 0

    return output_data


def invert(input_list):
    """
    invert: IMG
        Binary operation to interchange 0 and 1 in a binary mask.
    """
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError

    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    output_data = np.zeros(input_list[0].shape)
    output_data[input_list[0] != 0] = 0
    output_data[input_list[0] == 0] = 1

    return output_data


def dilation(input_list):
    """
    dilation: IMG
        Binary morphological operation to spatially extend the values of an
        image to their neighbors.
    """
    if not len(input_list) == 2:
        logging.error('This operation only support two operands.')
        raise ValueError

    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    if not is_float(input_list[1]):
        logging.error('Second input must be float/int for this operation.')
        raise ValueError

    return binary_dilation(input_list[0], iterations=int(input_list[1]))


def erosion(input_list):
    """
    erosion: IMG
        Binary morphological operation to spatially shrink the volume contained
        in a binary image.
    """
    if not len(input_list) == 2:
        logging.error('This operation only support two operands.')
        raise ValueError

    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    if not is_float(input_list[1]):
        logging.error('Second input must be float/int for this operation.')
        raise ValueError

    return binary_erosion(input_list[0], iterations=int(input_list[1]))


def closing(input_list):
    """
    closing: IMG
        Binary morphological operation, dilation followed by an erosion.
    """
    if not len(input_list) == 2:
        logging.error('This operation only support two operands.')
        raise ValueError

    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    if not is_float(input_list[1]):
        logging.error('Second input must be float/int for this operation.')
        raise ValueError

    return binary_closing(input_list[0], iterations=int(input_list[1]))


def opening(input_list):
    """
    opening: IMG
        Binary morphological operation, erosion followed by an dilation.
    """
    if not len(input_list) == 2:
        logging.error('This operation only support two operands.')
        raise ValueError

    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    if not is_float(input_list[1]):
        logging.error('Second input must be float/int for this operation.')
        raise ValueError

    return binary_opening(input_list[0], iterations=int(input_list[1]))


def gaussian_blur(input_list):
    """
    blur: IMG
        Apply a gaussian blur to a single image.
    """
    if not len(input_list) == 2:
        logging.error('This operation only support two operands.')
        raise ValueError

    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    if not is_float(input_list[1]):
        logging.error('Second input must be float/int for this operation.')
        raise ValueError

    return gaussian_filter(input_list[0], sigma=input_list[1])
