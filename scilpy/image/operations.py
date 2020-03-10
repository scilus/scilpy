# -*- coding: utf-8 -*-

from copy import copy
import logging

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import (binary_closing, binary_dilation,
                                      binary_erosion, binary_opening)


def get_array_operations_doc():
    return """
    lower_threshold: IMG THRESHOLD
        All values below the threshold will be set to zero.
        All values above the threshold will be set to one.
    upper_threshold: IMG THRESHOLD
        All values below the threshold will be set to one.
        All values above the threshold will be set to zero.
    lower_clip: IMG THRESHOLD
        All values below the threshold will be set to threshold.
    upper_clip: IMG THRESHOLD
        All values above the threshold will be set to threshold.
    absolute_value: IMG
        All negative values will become positive.
    round: IMG
        Round all decimal values to the closest integer.
    ceil: IMG
        Ceil all decimal values to the next integer.
    floor: IMG
        Floor all decimal values to the previous integer.
    normalize_sum: IMG
        Normalize the image so the sum of all values is one.
    normalize_max: IMG
        Normalize the image so the maximum value is one.
    convert: IMG
        Perform no operation, but simply change the data type.
    addition: IMGs
        Add multiple images together.
    subtraction: IMG_1 IMG_2
        Subtract two images together.
    multiplication: IMGs
        Multiply multiple images together (danger of underflow and overflow)
    division: IMG_1 IMG_2
        Divide two images together (danger of underflow and overflow)
    mean: IMGs
        If a single 4D image is provided, average along the last dimension.
    std: IMGs
        Compute the standard deviation average of multiple images.
        If a single 4D image is provided, compute the STD along the last
        dimension.
    union: IMGs
        Binary operation to keep voxels that are in any file.
    intersection: IMGs
        Binary operation to keep the voxels that are present in all files.
    difference: IMG_1 IMG_2
        Binary operation to keep voxels from the first file that are not in
        the second file.
    invert: IMG
        Binary operation to interchange 0 and 1 in a binary mask.
    """


def get_image_operations_doc():
    return """
    dilation: IMG
        Binary morphological operation to spatially expand the values of an
        image to their neighbors.
    erosion: IMG
        Binary morphological operation to spatially shrink the values of an
        image.
    closing: IMG
        Binary morphological operation, dilation followed by an erosion.
    opening: IMG
        Binary morphological operation, erosion followed by an dilation.
    blur: IMG
        Apply a gaussian blur to a single image.
    """

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


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
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    return np.abs(input_list[0])


def around(input_list):
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    return np.round(input_list[0])


def ceil(input_list):
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    return np.ceil(input_list[0])


def floor(input_list):
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    return np.floor(input_list[0])


def normalize_sum(input_list):
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    return copy(input_list[0]) / np.sum(input_list[0])


def normalize_max(input_list):
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    return copy(input_list[0]) / np.max(input_list[0])


def convert(input_list):
    if not len(input_list) == 1:
        logging.error('This operation only support one operand.')
        raise ValueError
    if not isinstance(input_list[0], np.ndarray):
        logging.error('The input must be an array for this operation.')
        raise ValueError

    return copy(input_list[0])


def addition(input_list):
    if not len(input_list) > 1:
        logging.error('This operation requires at least two operands.')
        raise ValueError

    ref_array = find_array(input_list)

    output_data = np.zeros(ref_array.shape)
    for data in input_list:
        output_data += data

    return output_data


def subtraction(input_list):
    if not len(input_list) == 2:
        logging.error('This operation only support two operands.')
        raise ValueError

    _ = find_array(input_list)

    return input_list[0] - input_list[1]


def multiplication(input_list):
    if not len(input_list) > 1:
        logging.error('This operation requires at least two operands.')
        raise ValueError

    _ = find_array(input_list)

    output_data = input_list[0]
    for data in input_list[1:]:
        output_data *= data

    return output_data


def division(input_list):
    if not len(input_list) == 2:
        logging.error('This operation only support two operands.')
        raise ValueError

    ref_array = find_array(input_list)

    output_data = np.zeros(ref_array.shape)
    output_data[input_list[1] > 0] = input_list[0][input_list[1] > 0] \
        / input_list[1][input_list[1] > 0]
    return output_data


def mean(input_list):
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
    output_data = addition(input_list)
    output_data[output_data != 0] = 1

    return output_data


def intersection(input_list):
    output_data = multiplication(input_list)
    output_data[output_data != 0] = 1

    return output_data


def difference(input_list):
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
